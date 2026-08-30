class Node:
    def __init__(self, l, r):
        self.l = l
        self.r = r
        self.cnt = 0
        self.interlen = 0
        self.maxgap = 0
        self.lgap = 0
        self.rgap = 0

class SegmentTree:
    def __init__(self, ys):
        self.ys = ys
        self.n = len(ys) - 1
        self.nodes = [None] * (10*self.n)
        self._build(1, 0, self.n - 1)
    
    def _build(self, idx, l, r):
        node = Node(self.ys[l], self.ys[r+1])
        node.lgap = node.rgap = node.maxgap = node.r - node.l
        self.nodes[idx] = node
        if l == r:
            return
        mid = (l + r) // 2
        self._build(idx * 2, l, mid)
        self._build(idx * 2 + 1, mid + 1, r)
    
    def _push_up(self, idx, l, r):
        node = self.nodes[idx]
        # （left_node, right_node)
        # print(idx, len(self.nodes))
        left_node = self.nodes[idx*2]
        right_node = self.nodes[idx*2 + 1]
        if node.cnt > 0:
            node.interlen = node.r - node.l
            node.lgap = node.rgap = node.maxgap = 0
        else:
            if l == r:
                node.interlen = 0
                node.lgap = node.rgap = node.maxgap = node.r - node.l
            else:
                node.interlen = left_node.interlen + right_node.interlen
                node.lgap = left_node.lgap
                if left_node.cnt == 0 and left_node.maxgap == left_node.r - left_node.l:
                    node.lgap = left_node.r - left_node.l + right_node.lgap
                node.rgap = right_node.rgap
                if right_node.cnt == 0 and right_node.maxgap == right_node.r - right_node.l:
                    node.rgap = right_node.r - right_node.l + left_node.rgap
                node.maxgap = max(left_node.maxgap, right_node.maxgap, left_node.rgap + right_node.lgap)
    
    def update(self, y1, y2, val, idx=1, l=0, r=None):
        if r is None:
            r = self.n - 1
        node = self.nodes[idx]
        #print("node idx: ", idx, ", node.l: ", node.l, ", node.r: ", node.r, ", y1: ", y1, ", y2: ", y2)
        if y1 <= node.l and node.r <= y2:
            #print("node idx: ", idx, ", node.l: ", node.l, ", node.r: ", node.r)
            node.cnt += val
            self._push_up(idx, l, r)
            return
        mid_idx = (l + r) // 2
        left_node = self.nodes[idx * 2]
        if y1 < left_node.r:
            self.update(y1, min(y2, left_node.r), val, idx * 2, l, mid_idx)
        right_node = self.nodes[idx * 2 + 1]
        if y2 > right_node.l:
            self.update(max(y1, right_node.l), y2, val, idx * 2 + 1, mid_idx + 1, r)
        self._push_up(idx, l, r)
        # print("y1: ", y1, ", y2: ", y2)
        # print("node idx: ", idx, ", node.l: ", node.l, ", node.r: ", node.r, ", node.maxgap: ", node.maxgap, ", node.lgap: ", node.lgap, ", node.rgap: ", node.rgap)
        # print("right node idx: ", idx * 2 + 1, ", node.l: ", right_node.l, ", node.r: ", right_node.r, ", node.maxgap: ", self.nodes[idx * 2 + 1].maxgap, "node.maxlap: ", self.nodes[idx * 2 + 1].lgap, "node.rlap: ", self.nodes[idx * 2 + 1].rgap)
        # print("left node idx: ", idx * 2, ", node.l: ", left_node.l, ", node.r: ", left_node.r, ", node.maxgap: ", self.nodes[idx * 2].maxgap, ", node.lgap: ", self.nodes[idx * 2].lgap, ", node.rgap: ", self.nodes[idx * 2].rgap)