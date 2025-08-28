from PIL import Image
import requests
from transformers import (
    VisionTextDualEncoderModel,
    VisionTextDualEncoderProcessor,
    AutoImageProcessor,
    AutoTokenizer,
)

if __name__ == "__main__":
    model_path = "./ckpt"

    photo_paths = [
        "data/eval/5850229113_4fe05d5265_z.jpg",
        "data/eval/000000039769.jpg",
    ]

    text_list = ["a photo of a cat", "a photo of a dog"]

    #################################零样本分类任务#################################
    # 加载模型和处理器
    processor = VisionTextDualEncoderProcessor.from_pretrained(model_path)
    model = VisionTextDualEncoderModel.from_pretrained(model_path)

    images = [Image.open(p).convert("RGB") for p in photo_paths]

    # Process inputs
    inputs = processor(
        text=text_list,
        images=images,
        return_tensors="pt",
        padding=True,
    )

    outputs = model(**inputs)
    logits_per_text = outputs.logits_per_text
    probs = logits_per_text.softmax(dim=1)
    print(probs)

    photo_index = probs.argmax(dim=-1)
    for i, index in enumerate(photo_index):
        print(f"与文本：【{text_list[i]}】最相似的图片是：【{photo_paths[index]}】")
