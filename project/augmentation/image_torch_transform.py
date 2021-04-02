import torchvision.transforms as transforms


class TorchAug:
    def __init__(self, height=512, width=512):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomResizedCrop(size=(height, width)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

    def __call__(self, image):
        return self.transform(image)