import cv2
from pylab import *

gray_saber = cv2.imread(r'D:\programming_project\image_fusion_data\LytroDataset\lytro-04-A.jpg')

# gray_saber = cv2.cvtColor(saber, cv2.COLOR_RGB2GRAY)
# gray_saber = cv2.resize(gray_saber, (200, 200))


def LaplaceOperator(roi, operator_type):
    if operator_type == "fourfields":
        laplace_operator = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    elif operator_type == "eightfields":
        laplace_operator = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]])
    else:
        raise ("type Error")
    result = np.abs(np.sum(roi * laplace_operator))
    return result


def LaplaceAlogrithm(image, operator_type):
    new_image = np.zeros(image.shape)
    print(new_image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    print(image.shape)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = LaplaceOperator(image[i - 1:i + 2, j - 1:j + 2], operator_type)
    new_image = new_image * (255 / np.max(image))
    return new_image.astype(np.uint8)


def noisy(noise_typ, image):
    if noise_typ == "gauss":
        row, col, ch = image.shape
        mean = 0
        var = 0.1
        sigma = var ** 0.5
        gauss = np.random.normal(mean, sigma, (row, col, ch))
        gauss = gauss.reshape(row, col, ch)
        noisy = image + gauss
        return noisy
    elif noise_typ == "s&p":
        row, col, ch = image.shape
        s_vs_p = 0.5
        amount = 0.004
        out = np.copy(image)
        num_salt = np.ceil(amount * image.size * s_vs_p)
        coords = [np.random.randint(0, i - 1, int(num_salt))
                  for i in image.shape]
        out[coords] = 1
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
        out[coords] = 0
        return out
    elif noise_typ == "poisson":
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        noisy = np.random.poisson(image * vals) / float(vals)
        return noisy
    elif noise_typ == "speckle":
        row, col, ch = image.shape
        gauss = np.random.randn(row, col, ch)
        gauss = gauss.reshape(row, col, ch)
        noisy = image + image * gauss
        return noisy


plt.subplot(121)
plt.title("fourfields")
plt.imshow(LaplaceAlogrithm(gray_saber, "fourfields"), cmap="binary")
plt.axis("off")
plt.subplot(122)
plt.title("eightfields")
plt.imshow(LaplaceAlogrithm(gray_saber, "eightfields"), cmap="binary")
plt.axis("off")
plt.show()


cv2.imshow('lap4',(LaplaceAlogrithm(gray_saber, "fourfields")))
cv2.waitKey()


def RobertsOperator(roi):
    operator_first = np.array([[-1, 0], [0, 1]])
    operator_second = np.array([[0, -1], [1, 0]])
    return np.abs(np.sum(roi[1:, 1:] * operator_first)) + np.abs(np.sum(roi[1:, 1:] * operator_second))


def RobertsAlogrithm(image):
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0]):
        for j in range(1, image.shape[1]):
            image[i, j] = RobertsOperator(image[i - 1:i + 2, j - 1:j + 2])
    return image[1:image.shape[0], 1:image.shape[1]]

#
# Robert_saber = RobertsAlogrithm(gray_saber)
# plt.imshow(Robert_saber, cmap="binary")
# plt.axis("off")
# plt.show()
#
# cv2.imshow('robert',Robert_saber)
# cv2.waitKey()


def SobelOperator(roi, operator_type):
    if operator_type == "horizontal":
        sobel_operator = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
    elif operator_type == "vertical":
        sobel_operator = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    else:
        raise ("type Error")
    result = np.abs(np.sum(roi * sobel_operator))
    return result


def SobelAlogrithm(image, operator_type):
    new_image = np.zeros(image.shape)
    image = cv2.copyMakeBorder(image, 1, 1, 1, 1, cv2.BORDER_DEFAULT)
    for i in range(1, image.shape[0] - 1):
        for j in range(1, image.shape[1] - 1):
            new_image[i - 1, j - 1] = SobelOperator(image[i - 1:i + 2, j - 1:j + 2], operator_type)
    new_image = new_image * (255 / np.max(image))
    return new_image.astype(np.uint8)


plt.subplot(121)
plt.title("horizontal")
plt.imshow(SobelAlogrithm(gray_saber, "horizontal"), cmap="binary")
plt.axis("off")
plt.subplot(122)
plt.title("vertical")
plt.imshow(SobelAlogrithm(gray_saber, "vertical"), cmap="binary")
plt.axis("off")
plt.show()
