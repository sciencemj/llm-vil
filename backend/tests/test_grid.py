from llm_vil.world.grid import TileGrid, facing_toward


def test_astar_routes_around_walls_deterministically():
    g = TileGrid(6, 5, blocked={(2, 0), (2, 1), (2, 2), (2, 3)})
    p1 = g.astar((0, 0), [(4, 0)])
    p2 = g.astar((0, 0), [(4, 0)])
    assert p1 == p2 and p1 is not None
    assert p1[-1] == (4, 0)
    assert all(g.walkable(t) for t in p1)
    assert (2, 4) in p1
    for a, b in zip([(0, 0), *p1], p1, strict=False):
        assert abs(a[0] - b[0]) + abs(a[1] - b[1]) == 1


def test_astar_unreachable_and_trivial():
    g = TileGrid(3, 3, blocked={(1, 0), (1, 1), (1, 2)})
    assert g.astar((0, 0), [(2, 2)]) is None
    assert g.astar((0, 0), [(0, 0)]) == []


def test_nearest_walkable_and_los():
    g = TileGrid(5, 5, blocked={(2, 2)})
    assert g.nearest_walkable((2, 2)) in {(3, 2), (2, 3), (1, 2), (2, 1)}
    assert not TileGrid.line_of_sight((0, 2), (4, 2), {(2, 2)})
    assert TileGrid.line_of_sight((0, 0), (4, 0), {(2, 2)})


def test_facing_snaps_to_45_degrees():
    assert facing_toward((0, 0), (1, 0)) == 0
    assert facing_toward((0, 0), (0, 1)) == 90
    assert facing_toward((0, 0), (-1, 0)) == 180
    assert facing_toward((0, 0), (1, 1)) == 45
    assert facing_toward((0, 0), (0, 0), default=270) == 270
