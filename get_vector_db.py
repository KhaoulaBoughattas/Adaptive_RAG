import json
from qdrant_client import QdrantClient
from qdrant_client.http.models import Distance, VectorParams, PointStruct
import httpx

# ------------------------------------------------------------------------------
# 1️⃣ Charger les données avec les embeddings
# ------------------------------------------------------------------------------
input_file = "data/chunks_with_embeddings.json"

with open(input_file, "r", encoding='utf-8') as f:
    data = json.load(f)

# ------------------------------------------------------------------------------
# 2️⃣ Connection Qdrant avec augmentation du timeout
# ------------------------------------------------------------------------------
client = QdrantClient(
    host='localhost',
    port=6333,
    timeout=600  # Timeout en secondes (600 secondes = 10 minutes)
)

# ------------------------------------------------------------------------------
# 3️⃣ Création ou recréation de la collection
# ------------------------------------------------------------------------------
collection_name = "psybot-embedding"

# Supprimer si elle existe déjà
if client.collection_exists(collection_name):
    print(f"⚡ La collection {collection_name} existe déjà. Suppression...")
    client.delete_collection(collection_name)

print(f"🟣 Création de la collection {collection_name}")
client.create_collection(
    collection_name=collection_name,
    vectors_config=VectorParams(size=1024, distance=Distance.COSINE)  # 1024 = dimension de bge-m3
)

# ------------------------------------------------------------------------------
# 4️⃣ Insertion des points par lots
# ------------------------------------------------------------------------------
batch_size = 500  # Taille des lots pour éviter les timeout
points = []

# Construire les points à insérer
for idx, item in enumerate(data):
    points.append(
        PointStruct(
            id=idx,
            vector=item['embedding'], 
            payload={"page_content": item['page_content']} 
        )
    )

print(f"➡ Insertion de {len(points)} points dans Qdrant")

# Insérer les points par petits lots
for i in range(0, len(points), batch_size):
    batch = points[i:i+batch_size]
    try:
        client.upsert(
            collection_name=collection_name,
            points=batch
        )
        print(f"✅ {len(batch)} points insérés avec succès.")
    except Exception as e:
        print(f"❌ Erreur lors de l'insertion des points : {e}")

print("✅ Qdrant est prêt et opérationnel.")
