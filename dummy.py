import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
from torch.nn import TransformerEncoder, TransformerEncoderLayer


class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, pos_weight=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits, targets):
        pb = torch.sigmoid(logits)              
        ce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        p_t = pb * targets + (1 - pb) * (1 - targets)
        focal_factor = (1 - p_t) ** self.gamma
        loss = self.alpha * focal_factor * ce_loss
        return loss.mean()


class RBMLayer(nn.Module):
    def __init__(self, n_visible, n_hidden, momentum=0.9):
        super(RBMLayer, self).__init__()
        self.W = nn.Parameter(torch.empty(n_hidden, n_visible))
        nn.init.kaiming_uniform_(self.W, a=0.01)  
        self.v_bias = nn.Parameter(torch.zeros(n_visible))
        self.h_bias = nn.Parameter(torch.zeros(n_hidden))

        self.W_mom = torch.zeros_like(self.W)
        self.vb_mom = torch.zeros_like(self.v_bias)
        self.hb_mom = torch.zeros_like(self.h_bias)
        self.momentum = momentum

    def sample_h(self, v):
        p_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return torch.bernoulli(p_h), p_h

    def sample_v(self, h):
        p_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return torch.bernoulli(p_v), p_v

    def sample_hidden(self, v):
        return torch.sigmoid(F.linear(v, self.W, self.h_bias))

    def contrastive_divergence(self, v, lr=1e-4, k=1):
        v0 = v.detach()
        vk = v0
        for _ in range(k):
            h_k, _ = self.sample_h(vk)
            vk, _ = self.sample_v(h_k)
            vk = vk.detach()

        p_h0 = torch.sigmoid(F.linear(v0, self.W, self.h_bias))
        p_hk = torch.sigmoid(F.linear(vk, self.W, self.h_bias))

        dW = (p_h0.t() @ v0 - p_hk.t() @ vk) / v0.size(0)
        dvb = (v0 - vk).mean(0)
        dhb = (p_h0 - p_hk).mean(0)

        self.W_mom = self.momentum * self.W_mom + lr * dW
        self.vb_mom = self.momentum * self.vb_mom + lr * dvb
        self.hb_mom = self.momentum * self.hb_mom + lr * dhb

        with torch.no_grad():
            self.W.add_(self.W_mom)
            self.v_bias.add_(self.vb_mom)
            self.h_bias.add_(self.hb_mom)
        return ((v0 - vk) ** 2).mean().item()


class DBN(nn.Module):
    def __init__(self, n_visible, hidden_sizes=[512, 256, 128], 
                 use_focal=False, focal_alpha=1.0, focal_gamma=2.0):
        super(DBN, self).__init__()
        # 1) RBM-layers (pretraining)
        self.rbms = nn.ModuleList([
            RBMLayer(n_visible if i == 0 else hidden_sizes[i - 1], hidden_sizes[i])
            for i in range(len(hidden_sizes))
        ])
        self.token_dim = 16  
        self.attn_embed = nn.Linear(1, self.token_dim)
        encoder_layer = TransformerEncoderLayer(
        d_model=self.token_dim,
        nhead=2,
        dim_feedforward=64,
        dropout=0.1,
        activation='gelu',
        batch_first=True
    )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=1)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_sizes[-1] * 16, 512),  
            nn.BatchNorm1d(512),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.5),

            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.4),

            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.3),

            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2),

            nn.Linear(64, 1)
        )
        self.use_focal = use_focal
        if self.use_focal:
            self.focal_alpha = focal_alpha
            self.focal_gamma = focal_gamma
            self.criterion = None
        else:
            self.criterion = None 

    def forward(self, x):
        for rbm in self.rbms:
            _, x = rbm.sample_h(x)
        batch_size = x.size(0)
        hid_dim = x.size(1) 
        x_unsq = x.unsqueeze(2)  
        x_emb = self.attn_embed(x_unsq)  
        x_transformed = self.transformer(x_emb)  
        x_flat = x_transformed.contiguous().view(batch_size, -1)  
        out = self.classifier(x_flat)
        return out
        
    def extract_features(self, x):
        for rbm in self.rbms:
            x = rbm.sample_hidden(x)
        return x

    def train_model(self,
                    X_train, y_train,
                    val_data=None,
                    epochs=100,
                    batch_size=256,
                    lr=1e-4,
                    weight_decay=1e-4,
                    clip_grad=1.0,
                    patience=20):
       
        device = X_train.device

        # Pre-training RBM
        logging.info("=== Starting DBN pre-training (layer-wise RBM) ===")
        x = X_train
        for idx, rbm in enumerate(self.rbms):
            logging.info(f"--> Pre-training RBM layer {idx+1}/{len(self.rbms)}")
            for ep in range(epochs):
                perm = torch.randperm(x.size(0), device=device)
                running_loss = 0.0
                for i in range(0, x.size(0), batch_size):
                    batch = x[perm[i:i+batch_size]]
                    cd_loss = rbm.contrastive_divergence(batch, lr=lr, k=1)
                    running_loss += cd_loss
                if ep % 10 == 0:
                    avg_loss = running_loss / (x.size(0)/batch_size)
                    logging.info(f"  - RBM{idx+1} Epoch {ep:3d}: ReconLoss={avg_loss:.6f}")
            with torch.no_grad():
                _, x = rbm.sample_h(x)

        # Fine-tuning supervised
        logging.info("=== Starting DBN fine-tuning (supervised) ===")
        neg_count = (y_train == 0).sum().item()
        pos_count = (y_train == 1).sum().item()
        pos_weight = torch.tensor([neg_count / pos_count], device=device)

        # Inisialisasi criterion
        if self.use_focal:
            self.criterion = FocalLoss(alpha=self.focal_alpha, gamma=self.focal_gamma, pos_weight=pos_weight)
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        optimizer = torch.optim.AdamW(self.parameters(), lr=lr, weight_decay=weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=3, min_lr=1e-6
        )

        self.train_losses, self.train_accs = [], []
        self.val_losses, self.val_accs = [], []

        best_val_loss = float('inf')
        wait = 0

        for ep in range(epochs):
            self.train()
            perm = torch.randperm(X_train.size(0), device=device)
            total_loss, total_correct, total_samples = 0.0, 0, 0

            for i in range(0, X_train.size(0), batch_size):
                idxs = perm[i:i+batch_size]
                xb = X_train[idxs]
                yb = y_train[idxs].unsqueeze(1).float()

                out = self.forward(xb)  # logit → [B,1]
                loss = self.criterion(out, yb)

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.parameters(), clip_grad)
                optimizer.step()

                total_loss += loss.item() * xb.size(0)
                preds = (torch.sigmoid(out) > 0.4).float()
                total_correct += (preds == yb).sum().item()
                total_samples += xb.size(0)

            train_loss = total_loss / total_samples
            train_acc = total_correct / total_samples
            self.train_losses.append(train_loss)
            self.train_accs.append(train_acc)

            # Validasi
            if val_data is not None:
                self.eval()
                X_val, y_val = val_data
                with torch.no_grad():
                    out_v = self.forward(X_val)
                    loss_v = self.criterion(out_v, y_val.unsqueeze(1).float()).item()
                    preds_v = (torch.sigmoid(out_v) > 0.4).float()
                    acc_v = (preds_v == y_val.unsqueeze(1).float()).float().mean().item()

                self.val_losses.append(loss_v)
                self.val_accs.append(acc_v)
                scheduler.step(loss_v)

                logging.info(
                    f"Epoch {ep:03d} | "
                    f"Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                    f"Val Loss={loss_v:.4f}, Acc={acc_v:.4f} | "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}"
                )

                if loss_v < best_val_loss:
                    best_val_loss = loss_v
                    wait = 0
                else:
                    wait += 1
                    if wait >= patience:
                        logging.info(f"Early stopping triggered at epoch {ep}")
                        break
            else:
                logging.info(
                    f"Epoch {ep:03d} | Train Loss={train_loss:.4f}, Acc={train_acc:.4f} | "
                    f"LR={optimizer.param_groups[0]['lr']:.6f}"
                )

        logging.info("=== Fine-tuning selesai ===")
        return {
            "train_losses": self.train_losses,
            "train_accs": self.train_accs,
            "val_losses": self.val_losses,
            "val_accs": self.val_accs
        }
