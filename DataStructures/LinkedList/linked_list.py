class Node:
    def __init__(self, data, next = None):
        self.data = data
        self.next = next

class LinkedList:
    def __init__(self, head=None):
        self.head = head

    def append(self, node):
        current = self.head
        while current is not None:
            current = current.next
        current.next = node
        node.next = None

if __name__=="__main__":
    head = Node(0)
    one = Node(1)
    two = Node(2)
    three = Node(3)

    head.next = one
    one.next = two
    two.next = three
    current = head
    while current is not None:
        print(current.data,end="")
        print("->",end="")
        current = current.next
    print("None")

