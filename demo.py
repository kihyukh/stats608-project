from graph import Graph

def demo_graph1():
    g = Graph(
        12,
        {
            (0, 1): 1,
            (0, 2): 1,
            (0, 3): 1,
            (0, 4): 1,
            (0, 5): 1,
            (1, 6): 1,
            (2, 6): 1,
            (3, 7): 0.5,
            (4, 8): 1,
            (5, 8): 1,
            (6, 9): 1,
            (7, 9): 1,
            (7, 10): 1,
            (8, 10): 1,
            (9, 11): 0.5,
            (10, 11): 1,
        },
        {
            0: [-1, 0],
            1: [-0.5, 1],
            2: [-0.5, 0.5],
            3: [-0.5, 0],
            4: [-0.5, -0.5],
            5: [-0.5, -1],
            6: [0, 0.5],
            7: [0, 0],
            8: [0, -0.5],
            9: [0.5, 0.25],
            10: [0.5, -0.25],
            11: [1, 0],
        }
    )
    return g

def demo_graph2():
    g = Graph(
        12,
        {
            (0, 1): 1,
            (0, 2): 1,
            (0, 3): 1,
            (0, 4): 0.5,
            (0, 5): 1,
            (1, 6): 1,
            (2, 6): 1,
            (3, 7): 0.5,
            (4, 8): 0.5,
            (5, 8): 1,
            (6, 9): 1,
            (7, 9): 1,
            (7, 10): 1,
            (8, 10): 0.5,
            (9, 11): 0.5,
            (10, 11): 1,
        },
        {
            0: [-1, 0],
            1: [-0.5, 1],
            2: [-0.5, 0.5],
            3: [-0.5, 0],
            4: [-0.5, -0.5],
            5: [-0.5, -1],
            6: [0, 0.5],
            7: [0, 0],
            8: [0, -0.5],
            9: [0.5, 0.25],
            10: [0.5, -0.25],
            11: [1, 0],
        }
    )
    return g
