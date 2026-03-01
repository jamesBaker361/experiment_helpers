import requests
import io
from PIL import Image
from datasets import Dataset, Features, Value, Image as HFImage
import os

# -------------------------
# 1️⃣ Download bulk metadata
# -------------------------
bulk = requests.get("https://api.scryfall.com/bulk-data").json()

download_url = next(
    item["download_uri"]
    for item in bulk["data"]
    if item["type"] == "unique_artwork"   # change to default_cards if you want every printing
)

print("Downloading bulk card data...")
cards = requests.get(download_url).json()

# -------------------------
# 2️⃣ Collect creature cards
# -------------------------
creature_cards = [
    card for card in cards
    if "Creature" in card.get("type_line", "")
]

print(f"Found {len(creature_cards)} creature cards")

# -------------------------
# 3️⃣ Download art as PIL Images
# -------------------------
images = []
names = []
mana_costs = []
types = []

for c,card in enumerate(creature_cards):
    urls = []

    if "image_uris" in card:
        urls.append(card["image_uris"]["art_crop"])
    elif "card_faces" in card:
        for face in card["card_faces"]:
            if "image_uris" in face:
                urls.append(face["image_uris"]["art_crop"])

    for url in urls:
        try:
            response = requests.get(url, timeout=10)
            img = Image.open(io.BytesIO(response.content)).convert("RGB")

            images.append(img)
            names.append(card.get("name", ""))
            mana_costs.append(card.get("mana_cost", ""))
            types.append(card.get("type_line", ""))

        except Exception as e:
            print("Failed:", card.get("name"), e)
            

print(f"Downloaded {len(images)} images")

# -------------------------
# 4️⃣ Build HuggingFace Dataset
# -------------------------
features = Features({
    "image": HFImage(),
    "name": Value("string"),
    "mana_cost": Value("string"),
    "type_line": Value("string"),
})

dataset = Dataset.from_dict(
    {
        "image": images,
        "name": names,
        "mana_cost": mana_costs,
        "type_line": types,
    },
    features=features,
)

# -------------------------
# 5️⃣ Save dataset locally
# -------------------------
dataset.push_to_hub("jlbaker361/mtg-creatures")

print("Saved dataset to mtg_creature_art_dataset/")