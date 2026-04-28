import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from flask import Flask, render_template, request, jsonify, send_from_directory

app = Flask(__name__)

# ==========================================
# 1. CONFIGURATION & CHEMINS INTELLIGENTS
# ==========================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, 'model')

# --- LOGIQUE DE RECHERCHE DU DATASET ---

# Option A : Portable (Le dossier est DANS le projet)
path_portable = os.path.join(BASE_DIR, 'img_align_celeba')

# Option B : "User Home" (Ton cas actuel)
# os.path.expanduser('~') renvoie automatiquement "C:/Users/roman" (ou le nom de l'utilisateur courant)
# On cherche donc dans : C:/Users/roman/FaceAttributes/img_align_celeba
user_home = os.path.expanduser('~')
path_user_home = os.path.join(user_home, 'FaceAttributes', 'img_align_celeba')

if os.path.exists(path_portable):
    IMAGES_DIR = path_portable
    print(f"✅ Mode Portable activé : {IMAGES_DIR}")
elif os.path.exists(path_user_home):
    IMAGES_DIR = path_user_home
    print(f"✅ Mode User Home activé : {IMAGES_DIR}")
else:
    # Fallback : on met le chemin portable par défaut pour éviter le crash au démarrage
    # (mais les images ne s'afficheront pas si le dossier n'existe pas)
    print("⚠️ ATTENTION : Dossier images introuvable (Ni dans le projet, ni dans User/FaceAttributes)")
    IMAGES_DIR = path_portable


PATH_RESNET_EMB  = os.path.join(MODEL_DIR, 'embeddings_resnet50.pt')
PATH_ATTRS_NAMES = os.path.join(MODEL_DIR, 'celeba_attributes.pt')
PATH_IMG_NAMES   = os.path.join(MODEL_DIR, 'celeba_image_names.pt')
PATH_MLP_WEIGHTS = os.path.join(MODEL_DIR, 'mlp_turbo.pth')
PATH_IMAGE_HEAD  = os.path.join(MODEL_DIR, 'resnet_turbo.pth') 

device = "cpu"

# ==========================================
# 2. ARCHITECTURES (Version Notebook 128->256)
# ==========================================
class AttributeMLP(nn.Module):
    def __init__(self, input_dim=40, embedding_dim=300):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, embedding_dim)
        )
    def forward(self, x):
        return F.normalize(self.net(x), dim=1)

class ImageHead(nn.Module):
    def __init__(self, input_dim=2048, embedding_dim=300):
        super().__init__()
        self.fc = nn.Linear(input_dim, embedding_dim)
    def forward(self, x):
        return F.normalize(self.fc(x), dim=1)

# ==========================================
# 3. CHARGEMENT
# ==========================================
print("⏳ Chargement des données...")

try:
    image_names = torch.load(PATH_IMG_NAMES)
    attrs_gt = torch.load(PATH_ATTRS_NAMES)
    
    # Dictionnaire pour accès rapide
    name_to_idx = {name: i for i, name in enumerate(image_names)}
    
except Exception as e:
    print(f"❌ Erreur chargement listes: {e}")
    exit()

all_attrs = [
    "5_o_Clock_Shadow", "Arched_Eyebrows", "Attractive", "Bags_Under_Eyes",
    "Bald", "Bangs", "Big_Lips", "Big_Nose", "Black_Hair", "Blond_Hair",
    "Blurry", "Brown_Hair", "Bushy_Eyebrows", "Chubby", "Double_Chin",
    "Eyeglasses", "Goatee", "Gray_Hair", "Heavy_Makeup", "High_Cheekbones",
    "Male", "Mouth_Slightly_Open", "Mustache", "Narrow_Eyes", "No_Beard",
    "Oval_Face", "Pale_Skin", "Pointy_Nose", "Receding_Hairline", "Rosy_Cheeks",
    "Sideburns", "Smiling", "Straight_Hair", "Wavy_Hair",
    "Wearing_Earrings", "Wearing_Hat", "Wearing_Lipstick",
    "Wearing_Necklace", "Wearing_Necktie", "Young"
]

mlp = AttributeMLP(input_dim=len(all_attrs), embedding_dim=300)
img_head = ImageHead(input_dim=2048, embedding_dim=300)

try:
    mlp.load_state_dict(torch.load(PATH_MLP_WEIGHTS, map_location=device))
    
    full_resnet = torch.load(PATH_IMAGE_HEAD, map_location=device)
    new_state = {}
    if "backbone.fc.weight" in full_resnet: 
        new_state["fc.weight"] = full_resnet["backbone.fc.weight"]
        new_state["fc.bias"] = full_resnet["backbone.fc.bias"]
    elif "fc.weight" in full_resnet: 
        new_state = full_resnet
    
    img_head.load_state_dict(new_state)
    print("✅ Modèles chargés.")
except Exception as e:
    print(f"❌ Erreur Modèles: {e}")
    exit()

mlp.eval()
img_head.eval()

print("⏳ Projection Base Images...")
try:
    data = torch.load(PATH_RESNET_EMB, map_location=device)
    if isinstance(data, dict):
        raw_emb = data['embeddings'].float()
    else:
        raw_emb = data.float()
        
    with torch.no_grad():
        db_emb = img_head(raw_emb)
    print(f"✅ Base prête: {db_emb.shape}")
except Exception as e:
    print(f"❌ Erreur Projection: {e}")

# ==========================================
# 4. ROUTES
# ==========================================

@app.route('/images/<path:filename>')
def serve_images(filename):
    return send_from_directory(IMAGES_DIR, filename)

@app.route("/")
def index():
    # Page d'accueil : on récupère les attributs des exemples
    sample_files = [
        "000009.jpg", "000109.jpg", "000155.jpg", "000170.jpg", 
        "000190.jpg", "000194.jpg", "000214.jpg", "000283.jpg"
    ]
    
    samples_data = []
    
    for fname in sample_files:
        if fname in name_to_idx:
            idx = name_to_idx[fname]
            attr_vec = attrs_gt[idx]
            active_list = [all_attrs[j] for j, v in enumerate(attr_vec) if v == 1]
            
            # 3 attributs max pour l'affichage
            display_str = ", ".join(active_list[:3]).replace('_', ' ')
            if len(active_list) > 3:
                display_str += "..."
                
            samples_data.append({
                "filename": fname,
                "attrs": display_str
            })
            
    return render_template("index.html", attributes=all_attrs, samples=samples_data)

@app.route("/search", methods=["POST"])
def search():
    try:
        req_data = request.json
        query_attrs = req_data.get("attributes", [])
        target_count = int(req_data.get("k", 5))

        print(f"\n🔍 Recherche: {query_attrs} | Demande: {target_count}")

        # 1. Vecteur Requête
        query_vec = torch.zeros(len(all_attrs))
        for a in query_attrs:
            if a in all_attrs:
                query_vec[all_attrs.index(a)] = 1
        
        # 2. Inférence MLP
        with torch.no_grad():
            q_emb = mlp(query_vec.unsqueeze(0))

        # 3. Calcul Distances
        dists = torch.cdist(q_emb, db_emb)
        K_SEARCH = 2000 # On cherche large
        topk = torch.topk(dists, K_SEARCH, largest=False)
        indices = topk.indices.squeeze(0).tolist()
        scores = topk.values.squeeze(0).tolist()
        
        results = []
        count_found = 0
        
        for i, idx in enumerate(indices):
            fname = image_names[idx]
            full_path = os.path.join(IMAGES_DIR, fname)
            
            # Vérification importante
            if os.path.exists(full_path):
                img_attr_vec = attrs_gt[idx]
                active_list = [all_attrs[j] for j, v in enumerate(img_attr_vec) if v == 1]
                matched_list = [a for a in query_attrs if a in active_list]
                
                results.append({
                    "filename": fname,
                    "score": f"{scores[i]:.4f}",
                    "active": active_list,
                    "matched": matched_list,
                    "match_count": len(matched_list),
                    "total_query": len(query_attrs)
                })
                
                count_found += 1
                if count_found >= target_count:
                    break
        
        return jsonify(results)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True, port=5000)