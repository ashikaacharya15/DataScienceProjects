from matplotlib import pyplot as io
import numpy as np
from PIL import Image
import random

koala_image = io.imread("Koala.jpg")
print (koala_image.shape)

penguin_image = io.imread("Penguins.jpg")
print(penguin_image.shape)


def my_k_means(name, image, k=10, n=5):
    reshaped_image = image.reshape(-1)
    clusters = []
    for i in range(k):
        clusters.append(random.randint(1, 255))

    for i in range(n):
        print("Iteration ", i, clusters)
        cluster_points = {}
        compressed_image = []
        distance = {}
        for point in reshaped_image:
            for center in clusters:
                distance[center] = (point - center) ** 2
            key = min(distance, key=distance.get)
            compressed_image.append(int(round(key)))
            value = []
            if key in cluster_points:
                value = cluster_points.get(key)
            value.append(point)
            cluster_points[key] = value

        clusters = []
        for center in cluster_points.keys():
            points = cluster_points.get(center)
            avg = np.mean(points)
            clusters.append(avg)

    compressed_image = np.asarray(compressed_image)

    compressed_image = compressed_image.reshape(768, 1024, 3)

    im = Image.fromarray((compressed_image * 255).astype(np.uint8))
    file_name = name + '_' + str(k) + '.png'
    im.save(file_name)
    io.imshow(im)


print("Compressing koala image with different values of k and with rep=5")
for k in [2, 5, 10, 15, 20]:
    print("Number of clusters: ", k)
    my_k_means('koala', koala_image, k)

print("Compressing penguin image with different values of k and with rep=5")
for k in [2, 5, 10, 15, 20]:
    print("Number of clusters: ", k)
    my_k_means('penguin', penguin_image, k)
