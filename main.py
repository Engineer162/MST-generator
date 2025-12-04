import random

def generate_graph(n):
    """
    Genererer en tilfældig, sammenhængende graf
    repræsenteret som en n x n adjacency matrix.
    """
    graph = [[0 for _ in range(n)] for _ in range(n)]

    # Garanter sammenhæng (lav først en kæde)
    for i in range(n - 1):
        w = random.randint(1, 10)
        graph[i][i + 1] = w
        graph[i + 1][i] = w

    # Tilføj ekstra kanter tilfældigt
    for i in range(n):
        for j in range(i + 2, n):
            if random.random() < 0.3:
                w = random.randint(1, 10)
                graph[i][j] = w
                graph[j][i] = w

    return graph


def print_graph(graph):
    print("Graf (adjacency matrix):")
    for row in graph:
        print(row)


if __name__ == "__main__":
    n = 6
    graph = generate_graph(n)
    print_graph(graph)
