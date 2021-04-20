import torchvision.transforms as transforms


class TorchAug:
    def __init__(self, height=512, width=512, gray_image=False, gray_prob=1, eraser_aug=False, eraser_prob=0.3):
        image_transform = [transforms.ToTensor(),
                           transforms.RandomResizedCrop(size=(height, width)),
                           transforms.RandomHorizontalFlip(0.5)]
        if eraser_aug:
            image_transform.append(transforms.RandomErasing(p=eraser_prob))
        if gray_image:
            image_transform.append(transforms.RandomGrayscale(p=gray_prob))
            image_transform.append(transforms.Normalize(mean=[0.45, 0.45, 0.45], std=[0.22, 0.22, 0.22]))
        else:
            image_transform.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

        self.transform = transforms.Compose(image_transform)

    def __call__(self, image):
        return self.transform(image)
