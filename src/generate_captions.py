import os
import base64
import pandas as pd
from tqdm import tqdm
from openai import OpenAI
from dotenv import load_dotenv

#initialize env vars + openai client
load_dotenv()
client = OpenAI()

#input output paths
image_dir = "data/raw/images"
output_file = "data/processed/glaze_captions.csv"

#make sure directory exists
os.makedirs(os.path.dirname(output_file), exist_ok=True)

captions = []

#loop thru images
for fname in tqdm(os.listdir(image_dir)):
    if fname.lower().endswith((".jpg", ".jpeg", ".png")):
        image_path = os.path.join(image_dir, fname)

        #convert image to base64 for API
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

        #call GPT-4o-mini to generate a visual caption
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert ceramic artist describing pottery glaze appearances. Write concise, vivid descriptions of surface color and finish."
                },
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "Describe this pottery glaze in one short, natural sentence (e.g., 'Matte beige with gray undertones.')."
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{image_data}"
                            }
                        }
                    ]
                }
            ],
            max_tokens=60,
        )

        caption = response.choices[0].message.content.strip()
        captions.append({"filename": fname, "visual_caption": caption})

#save captions to csv
df = pd.DataFrame(captions)
df.to_csv(output_file, index=False)
print(f"✅ Saved image captions to {output_file}")
