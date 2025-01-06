"""
Your task is to write a Python function that implements the k-Means clustering algorithm.
This function should take specific inputs and produce a list of final centroids.
k-Means clustering is a method used to partition n points into k clusters.
The goal is to group similar points together and represent each group by its center (called the centroid).

Function Inputs:
points: A list of points, where each point is a tuple of coordinates (e.g., (x, y) for 2D points)
k: An integer representing the number of clusters to form
initial_centroids: A list of initial centroid points, each a tuple of coordinates
max_iterations: An integer representing the maximum number of iterations to perform
Function Output:
A list of the final centroids of the clusters, where each centroid is rounded to the nearest fourth decimal.
"""
def k_means_clustering(
        points: list[tuple[float, float]],
        k: int,
        initial_centroids: list[tuple[float, float]],
        max_iterations: int
) -> list[tuple[float, float]]:
    assert k == len(initial_centroids)
    s = [None] * len(points)
    centroids = initial_centroids
    ndim = len(points[0])

    for _ in range(max_iterations):
        # Assignment step
        for n in range(len(points)):
            best = float("inf")
            for c in range(k):
                curr = 0
                for dim in range(ndim):
                    curr += (points[n][dim] - centroids[c][dim]) ** 2
                if curr < best:
                    s[n] = c
                    best = curr
        # Centroid update
        for c in range(k):
            center = [0.] * len(centroids[c])
            num = 0
            for n in range(len(points)):
                if s[n] == c:
                    for dim in range(ndim):
                        center[dim] += points[n][dim]
                    num += 1
            for dim in range(ndim):
                center[dim] /= num

            centroids[c] = center

    # round results
    for n in range(k):
        for dim in range(ndim):
            centroids[n][dim] = round(centroids[n][dim], 4)
    return centroids

if __name__ == '__main__':
    points = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 6)]
    initial_centroids = [(1, 2), (3, 4)]
    k = 2
    max_iterations = 10
    print(
        k_means_clustering(points, k, initial_centroids, max_iterations)
    )
    # [(1.5, 2.5), (4.5, 5.5)]

