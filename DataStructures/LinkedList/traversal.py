from linked_list import Node

def sum(head):
    current = head
    s = 0
    while current:
        s += current.data
        current = current.next
    return s

if __name__=="__main__":
    head = Node(0)
    one = Node(1)
    two = Node(2)
    three = Node(3)
    head.next = one
    one.next = two
    two.next = three

    print(sum(head))