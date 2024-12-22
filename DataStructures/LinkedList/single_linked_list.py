from seaborn.external.docscrape import header


class Node:
    def __init__(self, data, next=None):
        self.data = data
        self.next = next
    def __repr__(self):
        return f"{self.data} --> "

class LinkedList:
    def __init__(self, head=None):
        self.head = head

    def __len__(self):
        current = self.head
        len = 0
        while current:
            len += 1
            current = current.next
        return len

    def sum(self):
        current = self.head
        _sum = 0
        while current:
            _sum += current.data
            current = current.next
        return _sum

    def insert(self, node, idx:int):
        if not isinstance(node, Node):
            node = Node(node)
        c = 0
        prev = self.head
        while c < idx-1 and prev.next:
            prev = prev.next
            c += 1
        if idx == 0:
            node.next = self.head
            self.head = node
        else:
            node.next = prev.next
            prev.next = node

    def __repr__(self):
        current = self.head
        while current:
            print(current.data,end="")
            print("-->", end="")
            current = current.next
        return "None"
    

if __name__=="__main__":
    head = Node(0, None)
    one = Node(1, None)
    two = Node(2, None)
    three = Node(3, None)
    head.next = one
    one.next = two
    two.next = three

    ll = LinkedList(head)
    print(ll)
    print(len(ll))
    print(ll.sum())
    ll.insert(4, 2)
    print(ll)
    print(len(ll))
    print(ll.sum())
    ll.insert(5, 0)
    print(ll)
    print(len(ll))
    print(ll.sum())
