"""
2D Transformations
"""
import numpy as np
import matplotlib.pyplot as plt


# Translation
"""
x' = x+t 
x' = [I t]* x_
x_' =[I t]
     [0 1] * x_   
"""
def translation_1(img,t):
    m,n = img.shape[0],img.shape[1]
    img2 = img
    img2.flags.writeable = True
    for i in range(m):
        for j in range(n):
            print(img2[i][j])
            # img2[i] = img2[i] + t[0][0]
            # img2[j] = img2[i] + t[0][1]
    return img2

def translation_2(img,t):
    I = np.eye(2,2)
    t = np.array(t).T
    a = np.hstack((I,t))
    m,n = img.shape[0],img.shape[1]
    img2 = img
    img2.flags.writeable = True
    for i in range(m):
        for j in range(n):
            pass
            # img2[i][j] = a * img2[i][j].T

    return img2


if __name__ == "__main__":
    fig = plt.figure()
    img = plt.imread('./images/lena_gray.jpg')
    plt.subplot(131)
    plt.imshow(img)
    plt.subplot(132)
    t = [[20,40]]
    img2 = translation_1(img,t)
    plt.imshow(img2)
    plt.subplot(131)
    img3 = translation_2(img,t)
    plt.imshow(img3)

    plt.show()