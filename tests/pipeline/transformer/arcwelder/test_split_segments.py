from rayforge.core.ops import (
    MoveToCommand,
    LineToCommand,
    ArcToCommand,
    SetPowerCommand,
    DisableAirAssistCommand,
)
from rayforge.pipeline.transformer.arcwelder import split_into_segments


def _to_dict(item):
    if hasattr(item, "__iter__"):
        return [_to_dict(c) for c in item]
    return item.__dict__


def _pathcompare(one, two):
    return _to_dict(one) == _to_dict(two)


def test_split_empty_commands():
    assert split_into_segments([]) == []


def test_split_single_move():
    commands = [MoveToCommand((0, 0, 0))]
    assert _pathcompare(
        split_into_segments(commands), [[MoveToCommand((0, 0, 0))]]
    )


def test_split_move_and_line():
    commands = [MoveToCommand((0, 0, 0)), LineToCommand((1, 0, 0))]
    assert _pathcompare(
        split_into_segments(commands),
        [[MoveToCommand((0, 0, 0)), LineToCommand((1, 0, 0))]],
    )


def test_split_move_and_arc():
    commands = [
        MoveToCommand((0, 0, 0)),
        ArcToCommand((1, 0, 0), (1, 1), False),
        LineToCommand((2, 0, 0)),
        LineToCommand((3, 0, 0)),
    ]
    assert _pathcompare(
        split_into_segments(commands),
        [
            [MoveToCommand((0, 0, 0)), ArcToCommand((1, 0, 0), (1, 1), False)],
            [
                MoveToCommand((1, 0, 0)),
                LineToCommand((2, 0, 0)),
                LineToCommand((3, 0, 0)),
            ],
        ],
    )


def test_split_state_commands():
    commands = [
        SetPowerCommand(1000),
        MoveToCommand((0, 0, 0)),
        LineToCommand((1, 0, 0)),
        DisableAirAssistCommand(),
    ]
    assert _pathcompare(
        split_into_segments(commands),
        [
            [SetPowerCommand(1000)],
            [MoveToCommand((0, 0, 0)), LineToCommand((1, 0, 0))],
            [DisableAirAssistCommand()],
        ],
    )


def test_split_long_mixed_segment():
    commands = [
        MoveToCommand((0, 0, 0)),
        LineToCommand((1, 0, 0)),
        LineToCommand((2, 0, 0)),
        MoveToCommand((3, 0, 0)),
        LineToCommand((4, 0, 0)),
        ArcToCommand((5, 0, 0), (1, 1), False),
        LineToCommand((6, 0, 0)),
        MoveToCommand((7, 0, 0)),
        ArcToCommand((8, 0, 0), (1, 1), False),
        MoveToCommand((7, 0, 0)),
        ArcToCommand((8, 0, 0), (1, 1), False),
    ]
    assert _pathcompare(
        split_into_segments(commands),
        [
            [
                MoveToCommand((0, 0, 0)),
                LineToCommand((1, 0, 0)),
                LineToCommand((2, 0, 0)),
            ],
            [MoveToCommand((3, 0, 0)), LineToCommand((4, 0, 0))],
            [ArcToCommand((5, 0, 0), (1, 1), False)],
            [MoveToCommand((5, 0, 0)), LineToCommand((6, 0, 0))],
            [MoveToCommand((7, 0, 0)), ArcToCommand((8, 0, 0), (1, 1), False)],
            [MoveToCommand((7, 0, 0)), ArcToCommand((8, 0, 0), (1, 1), False)],
        ],
    )
