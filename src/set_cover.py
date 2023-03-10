# simple greedy set cover by http://www.martinbroadhurst.com/greedy-set-cover-in-python.html
def set_cover(universe, subsets):
    """Find a family of subsets that covers the universal set"""
    elements = set(e for s in subsets for e in s)
    # Check the subsets cover the universe
    if elements != universe:
        #return None
        # some trees are not covered
        print(len(list(universe-elements))," trees have not been covered")
    covered = set()
    cover = []
    # Greedily add the subsets with the most uncovered points
    while covered != elements:
        subset = max(subsets, key=lambda s: len(s - covered))
        cover.append(subset)
        covered |= subset
 
    return cover