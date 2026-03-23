import os
from PIL import Image
from torch.utils.data import Dataset
import gcsfs

class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, split='train', transform=None, dataset='faceforensics'):
        """
        root_dir: GCS bucket path e.g. gs://deepfake-detection-data-2026
                  OR local path for local data
        split: train / val / test
        dataset: faceforensics or celebdf
        """
        self.transform = transform
        self.samples = []
        self.use_gcs = root_dir.startswith('gs://')
        self.fs = gcsfs.GCSFileSystem() if self.use_gcs else None

        if dataset == 'faceforensics':
            split_dir = os.path.join(root_dir, 'faceforensics', 'dataset_processed_split', split)
            real_dir = os.path.join(split_dir, 'Real')
            self._add_samples(real_dir, label=0)
            for fake_type in ['Deepfakes', 'Face2Face', 'FaceSwap',
                              'FaceShifter', 'NeuralTextures', 'DeepFakeDetection']:
                fake_dir = os.path.join(split_dir, fake_type)
                self._add_samples(fake_dir, label=1)

        elif dataset == 'celebdf':
            split_name = split.capitalize()
            real_dir = os.path.join(root_dir, 'celebdf', 'Celeb_V2', split_name, 'real')
            fake_dir = os.path.join(root_dir, 'celebdf', 'Celeb_V2', split_name, 'fake')
            self._add_samples(real_dir, label=0)
            self._add_samples(fake_dir, label=1)

    def _add_samples(self, directory, label):
        exts = ('.jpg', '.jpeg', '.png')
        if self.use_gcs:
            clean = directory.replace('gs://', '')
            if self.fs.exists(clean):
                for fname in self.fs.ls(clean):
                    if fname.lower().endswith(exts):
                        self.samples.append((f'gs://{fname}', label))
        else:
            if os.path.exists(directory):
                for fname in os.listdir(directory):
                    if fname.lower().endswith(exts):
                        self.samples.append((os.path.join(directory, fname), label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        if self.use_gcs:
            with self.fs.open(path, 'rb') as f:
                image = Image.open(f).convert('RGB')
        else:
            image = Image.open(path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label
