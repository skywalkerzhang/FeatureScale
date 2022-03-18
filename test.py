from typing import (
    List,
)
from lintcode import (
    Interval,
)

"""
Definition of Interval:
class Interval(object):
    def __init__(self, start, end):
        self.start = start
        self.end = end
"""

class Solution:
    """
    @param a:
    @param query:
    @return:
    """
    def interval_x_o_r(self, a: List[int], query: List[Interval]) -> List[int]:
        res = []
        for q in query:
            t = 0
            for i in range(q.start, q.start + q.end):
                t ^= a[i]
            res.append(t)
        return res

