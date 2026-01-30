import os

import pytest
from _pytest.mark.expression import Expression


def pytest_addoption(parser):
    parser.addoption(
        "--runslow",
        action="store_true",
        default=False,
        help="run tests marked as slow",
    )


def pytest_configure(config):
    config.addinivalue_line("markers", "slow: mark test as slow")


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
):
    """
    Hook to modify collected test items.

    Parameters
    ----------
    config: pytest.Config
        pytest configuration
    items: list[pytest.Item]
        list of pytest.Item objects (i.e., collected tests)
    """
    selected_items = []
    deselected_items = []
    # Always run test if it's a single test invoked from vscode
    if len(items) == 1 and "vscode_pytest" in config.invocation_params.args:
        return
    if not config.getoption("--runslow"):
        skip_slow = pytest.mark.skip(reason="need --runslow option to run")
        for item in items:
            if "slow" in item.keywords:
                item.add_marker(skip_slow)
    # markexpr = config.option.markexpr  # <-- This is what was passed with -m
    markexpr = config.getoption("-m")  # <-- This is what was passed with -m
    if markexpr is not None:
        if markexpr == "not skip or not slow":
            marker_list = ["unit", "functional", "end-to-end"]
        else:
            # Compile the mark expression
            compiled_markexpr = Expression.compile(markexpr)
            # Return a sequence of markers that match
            marker_list = [
                mark
                for mark in [
                    "unit",
                    "functional",
                    "end_to_end",
                    "notebooks",
                    "docs",
                    "slow",
                ]
                if compiled_markexpr.evaluate(
                    lambda c: c == mark  # type: ignore # noqa
                )
            ]

    for item in items:
        # Check if all tests have a marker
        if not any(
            marker.name
            in ["unit", "functional", "end_to_end", "notebooks", "skip", "docs"]
            for marker in item.iter_markers()
        ):
            msg = (
                f"Test {item.nodeid} is missing a required marker"
                "(unit, functional, end_to_end, notebooks, slow, skip)"
            )
            raise pytest.UsageError(msg)

        # Skip slow in VSCode or skip test marked as skip
        if (
            item.get_closest_marker("slow") is not None
            and "vscode_pytest" in config.invocation_params.args
        ) or item.get_closest_marker("skip") is not None:
            deselected_items.append(item)
            continue

        selected_items.append(item)

    # Get test path directory
    test_data_path = os.getenv("ETDATASET_TEST_DATA_PATH")
    for item in selected_items:
        # Check if ETDATASET_TEST_DATA_PATH is set for tests
        # with require_test_data marker
        item_markers = [m.name for m in item.iter_markers()]
        if (
            (set(item_markers) & set(marker_list))
            and ("require_test_data" in item_markers)
            and test_data_path is None
        ):
            msg = (
                f"Test {item.nodeid} is marked with @pytest.mark.require_data, "
                "but the ETDATASET_TEST_DATA environment variable is not set."
            )
            raise pytest.UsageError(msg)

    # Modify items in-place
    items[:] = selected_items

    # Inform pytest of deselected items
    config.hook.pytest_deselected(items=deselected_items)
