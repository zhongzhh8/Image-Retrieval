import os

from scipy.io import loadmat
from torch.utils.data import Dataset
from PIL import Image

class CUB_200(Dataset):
    def __init__(self, root, train=True, transform_train=None,transform_test=None):
        super(CUB_200, self).__init__()
        self.root = root
        self.train = train
        self.transform_test = transform_test
        self.transform_train = transform_train
        self.classes_file = os.path.join(root, 'classes.txt')  # <class_id> <class_name>
        self.image_class_labels_file = os.path.join(root, 'image_class_labels.txt')  # <image_id> <class_id>
        self.images_file = os.path.join(root, 'images.txt')  # <image_id> <image_name>
        self.train_test_split_file = os.path.join(root, 'train_test_split.txt')  # <image_id> <is_training_image>
        self.bounding_boxes_file = os.path.join(root, 'bounding_boxes.txt')  # <image_id> <x> <y> <width> <height>

        self._train_ids = []
        self._test_ids = []
        self._image_id_label = {}
        self._train_path_label = []
        self._test_path_label = []

        self._train_test_split()
        self._get_id_to_label()
        self._get_path_label()

    def _train_test_split(self):

        for line in open(self.train_test_split_file):
            image_id, label = line.strip('\n').split()
            if label == '1':
                self._train_ids.append(image_id)
            elif label == '0':
                self._test_ids.append(image_id)
            else:
                raise Exception('label error')

    def _get_id_to_label(self):
        for line in open(self.image_class_labels_file):
            image_id, class_id = line.strip('\n').split()
            self._image_id_label[image_id] = class_id

    def _get_path_label(self):
        for line in open(self.images_file):
            image_id, image_name = line.strip('\n').split()
            label = self._image_id_label[image_id]
            if image_id in self._train_ids:
                self._train_path_label.append((image_name, label))
            else:
                self._test_path_label.append((image_name, label))

    def __getitem__(self, index):
        if self.train:
            image_name, label = self._train_path_label[index]
        else:
            image_name, label = self._test_path_label[index]
        image_path = os.path.join(self.root, 'images', image_name)
        img = Image.open(image_path)
        if img.mode == 'L':
            img = img.convert('RGB')
        label = int(label)
        if self.train :
            img = self.transform_train(img)
        else:
            img=self.transform_test(img)
        return img, label

    def __len__(self):
        if self.train:
            return len(self._train_ids)
        else:
            return len(self._test_ids)



class Stanford_Dogs(Dataset):
    '''
    Stanford_Dog Dataset for image retrieval
    '''

    def __init__(self, root, train=True,transform_train=None,transform_test=None):
        '''
        file: data root.
        if_train: to identify train set of test set.
        '''
        self.root = root
        self.train = train
        self.transform_test = transform_test
        self.transform_train = transform_train

        if self.train:
            self.images = [image[0][0] for image in loadmat(os.path.join(root, 'train_list.mat'))['file_list']]
            self.labels = [(int(image[0]) - 1) for image in loadmat(os.path.join(root, 'train_list.mat'))['labels']]
        else:
            self.images = [image[0][0] for image in loadmat(os.path.join(root, 'test_list.mat'))['file_list']]
            self.labels = [(int(image[0]) - 1) for image in loadmat(os.path.join(root, 'test_list.mat'))['labels']]

        if self.train:
            assert len(self.images) == len(self.labels) == 12000
        else:
            assert len(self.images) == len(self.labels) == 8580



    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        imagename = self.images[idx]
        label = self.labels[idx]
        image = Image.open(os.path.join(self.root, 'Images', imagename)).convert('RGB')

        if self.train :
            image = self.transform_train(image)
        else:
            image = self.transform_test(image)

        return image, label
