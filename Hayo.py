# Anime_Hayo
# ✅ Step 1: Install dependencies
!pip install -U pip
!pip install torch==2.1.0+cu118 torchvision==0.16.0+cu118 torchaudio==2.1.0 --extra-index-url https://download.pytorch.org/whl/cu118
!pip install tqdm opencv-python==4.7.0.72 color-transfer-py==0.0.4 numpy==1.26.4

# ✅ Step 2: Clone AnimeGAN repo
!git clone https://github.com/ptran1203/pytorch-animeGAN.git
%cd pytorch-animeGAN

# ✅ Step 3: Download dataset
!wget -O anime-gan.zip https://github.com/ptran1203/pytorch-animeGAN/releases/download/v1.0/dataset_v1.zip
!unzip -q anime-gan.zip

# ✅ Step 4: Apply edge smoothing to anime dataset (e.g., Hayao)
!python3 script/edge_smooth.py --data-dir dataset/Hayao --image-size 256
# ✅ Step 5: Start training
!python3 train.py \
  --anime_image_dir dataset/Hayao \
  --real_image_dir dataset/train_photo/ \
  --model v2 \
  --batch_size 8 \
  --amp \
  --init_epochs 30 \
  --exp_dir runs \
  --save_interval 1 \
  --gan_loss lsgan \
  --init_lr 1e-4 \
  --lr_g 2e-5 \
  --lr_d 4e-5 \
  --wadvd 300.0 \
  --wadvg 300.0 \
  --wcon 1.5 \
  --wgra 3.0 \
  --wcol 30.0 \
  --use_sn
