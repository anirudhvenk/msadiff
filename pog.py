class Node:
    def __init__(self, char):
        self.char = char
        self.edges = []

class POG:
    def __init__(self):
        self.nodes = []

    def add_sequence(self, sequence):
        if not self.nodes:
            # Initialize POG with the first sequence
            self.nodes = [Node(char) for char in sequence]
        else:
            # Align new sequence to the existing POG
            self.align_sequence(sequence)

    def align_sequence(self, sequence):
        # Dynamic programming matrix
        dp = [[0] * (len(self.nodes) + 1) for _ in range(len(sequence) + 1)]
        
        # Fill the DP matrix
        for i in range(1, len(sequence) + 1):
            for j in range(1, len(self.nodes) + 1):
                if sequence[i - 1] == self.nodes[j - 1].char:
                    dp[i][j] = dp[i - 1][j - 1] + 1
                else:
                    dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
        
        # Traceback to build the POG
        i, j = len(sequence), len(self.nodes)
        new_nodes = []
        while i > 0 and j > 0:
            if sequence[i - 1] == self.nodes[j - 1].char:
                new_nodes.append(self.nodes[j - 1])
                i -= 1
                j -= 1
            elif dp[i - 1][j] >= dp[i][j - 1]:
                new_nodes.append(Node(sequence[i - 1]))
                i -= 1
            else:
                new_nodes.append(self.nodes[j - 1])
                j -= 1
        
        # Add remaining nodes
        while i > 0:
            new_nodes.append(Node(sequence[i - 1]))
            i -= 1
        while j > 0:
            new_nodes.append(self.nodes[j - 1])
            j -= 1
        
        # Reverse to maintain order
        new_nodes.reverse()
        self.nodes = new_nodes

# Example usage
pog = POG()
pog.add_sequence("ACGT")
pog.add_sequence("ACGTT")
pog.add_sequence("ACGTA")

# Print the POG
for node in pog.nodes:
    print(node.char, end=" ")