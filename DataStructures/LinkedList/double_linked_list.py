from sympy.physics.units import current


class Node:
    def __init__(self, data, next=None, prev=None):
        self.data = data
        self.next = next
        self.prev = prev

    def __repr__(self):
        return f"{self.data} <--> "


class DoubleLinkedList:
    def __init__(
            self,
            head=None
    ):
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

    def insert(self, node, idx: int):
        if not isinstance(node, Node):
            node = Node(node)
        c = 0
        current = self.head
        while c < idx and current.next:
            current = current.next
            c += 1
        if idx == 0:
            node.next = self.head
            node.prev = self.head.prev
            self.head.prev = node
            self.head = node
        else:
            node.next = current
            node.prev = current.prev
            current.prev.next = node
            current.prev = node

    def remove(self ,idx):
        current = self.head
        c = 0
        while current and c<idx:
            current = current.next
            c += 1
        if not current:
            raise IndexError("Index out of range")
        current.prev.next = current.next
        current.next.prev = current.prev
        del current

    def __repr__(self):
        current = self.head
        while current:
            print(current.data, end="")
            print("<-->", end="")
            current = current.next
        return "None"


if __name__ == "__main__":
    head = Node(0)
    one = Node(1)
    two = Node(2)
    three = Node(3)
    head.next = one
    one.prev = head
    one.next = two
    two.prev = one
    two.next = three

    ll = DoubleLinkedList(head)
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
    ll.remove(2)
    print(ll)
    print(len(ll))
    print(ll.sum())
