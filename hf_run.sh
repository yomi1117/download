# 科学上网，设置代理
source /pfs/yangyuanming/set_proxy.sh

# #下载模型：openai/clip-vit-large-patch14
# python hf_downloader.py \
#   --repo-id openai/clip-vit-large-patch14 \
#   --repo-type model \
#   --local-dir /pfs/yangyuanming/code2/models/clip-vit-large-patch14 \

# # 下载模型:openai/clip-vit-large-patch14-336
# python hf_downloader.py \
#   --repo-id openai/clip-vit-large-patch14-336 \
#   --repo-type model \
#   --local-dir /pfs/yangyuanming/code2/models/clip-vit-large-patch14-336 \



# # 下载模型：yuvalkirstain/PickScore_v1
# python hf_downloader.py \
#   --repo-id yuvalkirstain/PickScore_v1 \
#   --repo-type model \
#   --local-dir /pfs/yangyuanming/code2/models/PickScore_v1 \

# 下载模型：CLIP-ViT-H-14-laion2B-s32B-b79K
python hf_downloader.py \
  --repo-id laion/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --repo-type model \
  --local-dir /pfs/yangyuanming/code2/models/CLIP-ViT-H-14-laion2B-s32B-b79K \
  --timeout 120


# # 下载模型：FLUX.1-dev
# python hf_downloader.py \
#   --repo-id black-forest-labs/FLUX.1-dev \
#   --repo-type model \
#   --local-dir /pfs/yangyuanming/code2/models/FLUX.1-dev \


# # 下载模型：仅下 safetensors 权重和 tokenizer
# python hf_downloader.py \
#   --repo-id meta-llama/Llama-3.1-8B \
#   --repo-type model \
#   --local-dir /pfs/yangyuanming/code2/models/Llama-3.1-8B \
#   --allow "*.safetensors" "tokenizer.*" "config.json"

# # 下载数据集：只下 parquet
# python hf_downloader.py \
#   --repo-id HuggingFaceH4/ultrachat_200k \
#   --repo-type dataset \
#   --allow "*.parquet"
