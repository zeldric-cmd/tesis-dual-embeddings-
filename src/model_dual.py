"""
Modelo CNN1D con embeddings DUALES para resolver el tradeoff warm/cold.

Idea central:
- Tabla CONGELADA (FastText): preserva semántica léxica -> sirve en cold-start.
- Tabla ENTRENABLE (init random): aprende co-ocurrencias -> sirve en warm-start.
- Combinación: sum / concat / gate (el gate aprende cuánto pesar cada una).

Técnica clave de entrenamiento (en training_dual.py):
- Cold-start dropout: con probabilidad p, se "apaga" la tabla entrenable de
  algunos productos durante el forward, simulando que son productos nuevos.
  Esto FUERZA al modelo a aprender a funcionar usando solo la semántica,
  que es lo que necesitará en cold-start real.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class DualEmbeddingCNN(nn.Module):
    """
    CNN1D multi-escala con dos tablas de embeddings (congelada + entrenable).

    Args:
        n_productos: número de productos del catálogo
        emb_dim: dimensión de CADA tabla (300)
        n_filters: filtros por bloque convolucional (256)
        dropout: dropout en bloques conv (0.3)
        seq_len: longitud de secuencia (10)
        combine_mode: "sum" | "concat" | "gate"
    """
    def __init__(self, n_productos, emb_dim=300, n_filters=256,
                 dropout=0.3, seq_len=10, combine_mode="gate"):
        super().__init__()
        self.n_productos = n_productos
        self.emb_dim = emb_dim
        self.n_filters = n_filters
        self.seq_len = seq_len
        self.combine_mode = combine_mode

        # Tabla CONGELADA (FastText)
        # padding_idx=0; se inicializa con init_frozen_embeddings()
        self.emb_frozen = nn.Embedding(n_productos + 1, emb_dim, padding_idx=0)

        # Tabla ENTRENABLE (init aleatoria pequeña)
        self.emb_trainable = nn.Embedding(n_productos + 1, emb_dim, padding_idx=0)
        nn.init.normal_(self.emb_trainable.weight, mean=0.0, std=0.01)
        with torch.no_grad():
            self.emb_trainable.weight[0].zero_()  # padding a cero

        # Dimensión efectiva del embedding combinado
        if combine_mode == "concat":
            self.emb_combinado_dim = emb_dim * 2   # 600
        else:  # sum, gate
            self.emb_combinado_dim = emb_dim       # 300

        # Gate: red pequeña que produce alpha en [0,1] por producto
        # alpha controla cuánto pesa la tabla semántica vs la entrenable.
        if combine_mode == "gate":
            self.gate = nn.Sequential(
                nn.Linear(emb_dim * 2, emb_dim),
                nn.ReLU(),
                nn.Linear(emb_dim, 1),
                nn.Sigmoid()
            )

        #Tres bloques convolucionales paralelos (k=3,5,7)
        self.conv3 = nn.Sequential(
            nn.Conv1d(self.emb_combinado_dim, n_filters, kernel_size=3, padding=1),
            nn.BatchNorm1d(n_filters), nn.ReLU(), nn.Dropout1d(dropout)
        )
        self.conv5 = nn.Sequential(
            nn.Conv1d(self.emb_combinado_dim, n_filters, kernel_size=5, padding=2),
            nn.BatchNorm1d(n_filters), nn.ReLU(), nn.Dropout1d(dropout)
        )
        self.conv7 = nn.Sequential(
            nn.Conv1d(self.emb_combinado_dim, n_filters, kernel_size=7, padding=3),
            nn.BatchNorm1d(n_filters), nn.ReLU(), nn.Dropout1d(dropout)
        )

        # Proyección final al espacio de comparación (emb_dim)
        # El perfil de usuario y los embeddings de candidatos se comparan aquí.
        self.projection = nn.Linear(n_filters * 3, emb_dim)


    def combinar(self, idx, cold_mask=None):
        """
        Combina las dos tablas para un tensor de índices.

        Args:
            idx: (batch, seq_len) índices de productos
            cold_mask: (batch, seq_len) bool, True = tratar como cold
                       (apagar tabla entrenable para ese producto).
                       Si None, no se apaga nada.

        Returns:
            emb_comb: (batch, seq_len, emb_combinado_dim)
        """
        e_sem = self.emb_frozen(idx)      # (B, L, D) semántico
        e_col = self.emb_trainable(idx)   # (B, L, D) colaborativo

        # Cold-start dropout: apagar la parte colaborativa donde cold_mask=True
        if cold_mask is not None:
            # cold_mask: (B, L) -> (B, L, 1)
            keep = (~cold_mask).unsqueeze(-1).float()
            e_col = e_col * keep

        if self.combine_mode == "sum":
            return e_sem + e_col
        elif self.combine_mode == "concat":
            return torch.cat([e_sem, e_col], dim=-1)
        elif self.combine_mode == "gate":
            # alpha en [0,1] por (producto): cuánto pesa lo semántico
            gate_in = torch.cat([e_sem, e_col], dim=-1)  # (B, L, 2D)
            alpha = self.gate(gate_in)                    # (B, L, 1)
            return alpha * e_sem + (1.0 - alpha) * e_col
        else:
            raise ValueError(f"combine_mode desconocido: {self.combine_mode}")

    def forward(self, x, cold_mask=None, return_activations=False):
        """
        Args:
            x: (batch, seq_len) índices del historial
            cold_mask: (batch, seq_len) bool opcional para cold-start dropout
        Returns:
            perfil: (batch, emb_dim) L2-normalizado
        """
        emb = self.combinar(x, cold_mask)   # (B, L, emb_comb_dim)
        emb = emb.permute(0, 2, 1)          # (B, emb_comb_dim, L)

        h3 = self.conv3(emb)
        h5 = self.conv5(emb)
        h7 = self.conv7(emb)

        activations = None
        if return_activations:
            activations = {'conv3': h3.detach(), 'conv5': h5.detach(), 'conv7': h7.detach()}

        g3 = h3.mean(dim=2)
        g5 = h5.mean(dim=2)
        g7 = h7.mean(dim=2)
        h = torch.cat([g3, g5, g7], dim=1)
        out = self.projection(h)
        out = F.normalize(out, dim=-1)

        if return_activations:
            return out, activations
        return out

    
    def init_frozen_embeddings(self, embedding_matrix):
        """Inicializa la tabla CONGELADA con FastText y la congela."""
        with torch.no_grad():
            self.emb_frozen.weight[1:] = torch.FloatTensor(embedding_matrix)
            self.emb_frozen.weight[0].zero_()
        self.emb_frozen.weight.requires_grad = False
        print(f"Tabla CONGELADA inicializada con FastText y congelada: {embedding_matrix.shape}")

    def get_candidate_embeddings(self, idx, cold_mask=None):
        """
        Embedding de productos CANDIDATOS para comparar con el perfil.

        Para WARM: usa ambas tablas (cold_mask=None).
        Para COLD: cold_mask=True -> solo tabla semántica (el producto nuevo
                   no tiene embedding entrenable útil).

        Args:
            idx: (N,) o (batch, N) índices de candidatos
            cold_mask: bool del mismo shape que idx
        Returns:
            embeddings combinados, L2-normalizados, listos para producto punto
        """
        # Asegurar forma (..., 1) para reutilizar combinar
        original_shape = idx.shape
        idx_flat = idx.reshape(1, -1)  # (1, M)
        cm = None
        if cold_mask is not None:
            cm = cold_mask.reshape(1, -1)
        emb = self.combinar(idx_flat, cm)  # (1, M, emb_comb_dim)
        emb = emb.squeeze(0)               # (M, emb_comb_dim)

        # OJO: para comparar con el perfil (que sale de projection -> emb_dim),
        # si combine_mode == concat los candidatos quedan en 2D dim.
        # En ese caso hay que proyectar también los candidatos. Lo resolvemos
        # en eval con un proyector dedicado. Para sum/gate ya están en emb_dim.
        return emb

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


if __name__ == "__main__":
    # Test rápido
    n = 47618
    for mode in ["sum", "concat", "gate"]:
        model = DualEmbeddingCNN(n_productos=n, combine_mode=mode)
        fake_emb = torch.randn(n, 300).numpy()
        model.init_frozen_embeddings(fake_emb)
        x = torch.randint(1, n+1, (4, 10))
        out = model(x)
        # Test con cold_mask
        cm = torch.zeros(4, 10, dtype=torch.bool)
        cm[:, :3] = True  # primeros 3 productos como "cold"
        out_cold = model(x, cold_mask=cm)
        print(f"[{mode}] params entrenables: {model.count_parameters():,} | "
              f"out: {tuple(out.shape)} | norm: {out.norm(dim=-1).mean():.3f}")
