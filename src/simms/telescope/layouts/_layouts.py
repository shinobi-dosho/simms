import glob
import os

from simms.utilities import load_yaml

thisdir = os.path.dirname(__file__)


def _per_antenna_telescope_names(arrayinfo, nant: int):
    """Per-antenna ``telescope_name`` list, defaulting to the array name.

    Accepts a scalar (broadcast to all antennas) or a per-antenna list, mirroring how
    ``size`` is handled so subarray selection can index it.
    """
    names = arrayinfo.get("telescope_name", "") or arrayinfo.get("name", "")
    if isinstance(names, str):
        return [names] * nant
    return list(names)


def simms_telescopes() -> dict:
    """
    Returns a dictionary of known array layouts
    """
    lays = map(str, glob.glob(f"{thisdir}/*.geodetic.yaml"))
    laysdict = {}
    for layout in lays:
        # Array name
        arrayinfo = load_yaml(layout)
        allants = list(arrayinfo["antnames"])
        all_locations = list(arrayinfo["antlocations"])
        anant = len(all_locations)
        ant_to_idx = {name: i for i, name in enumerate(allants)}

        allsizes = arrayinfo["size"]
        allsizes = [allsizes] * anant if isinstance(allsizes, float | int) else list(allsizes)

        alltelnames = _per_antenna_telescope_names(arrayinfo, anant)

        subarrays = arrayinfo.get("subarray", [])
        # add sub-arrays to database
        for subarray in subarrays:
            antnames = arrayinfo["subarray"][subarray]
            antlocations = []
            antsizes = []
            anttelnames = []
            for ant in antnames:
                idx = ant_to_idx[ant]
                antlocations.append(all_locations[idx])
                antsizes.append(allsizes[idx])
                anttelnames.append(alltelnames[idx])

            laysdict[subarray] = dict(
                centre=arrayinfo["centre"],
                antlocations=antlocations,
                antnames=antnames,
                size=antsizes,
                telescope_name=anttelnames,
                coord_sys=arrayinfo["coord_sys"],
                mount=arrayinfo["mount"],
                issubarray=True,
            )

        # add main layout, keyed by its declared name (falling back to the file's stem)
        lname = arrayinfo.get("name") or os.path.basename(layout).split(".")[0]
        laysdict[lname] = arrayinfo

    return laysdict


def resolve_layout(layout: str):
    """Resolve a layout name or a path to a layout YAML to ``(arrayinfo, path)``.

    ``path`` is the file the layout was read from, or ``None`` for an entry already built in
    :data:`SIMMS_TELESCOPES`. Resolving by name first is what makes a *named* subarray
    (``meerkat``, ``skamid-aa1``, ...) usable here: those are registered as telescopes in
    their own right, with their antennas already selected, and have no layout file of their
    own -- so building ``<thisdir>/<layout>.geodetic.yaml`` cannot find them, and neither can
    it find a user's layout file given by path.
    """
    if layout in SIMMS_TELESCOPES:
        return SIMMS_TELESCOPES[layout], None

    builtin = os.path.join(thisdir, f"{layout}.geodetic.yaml")
    if os.path.exists(builtin):
        return load_yaml(builtin), builtin

    if os.path.exists(layout):
        return load_yaml(layout), layout

    raise FileNotFoundError(
        f"Layout {layout!r} is neither a known telescope nor an existing layout file. "
        f"Known telescopes: {', '.join(sorted(SIMMS_TELESCOPES))}."
    )


def custom_telescopes(layout: str, subarray_list=None, subarray_range=None, subarray_file: str | None = None) -> dict:
    """
    Returns a dictionary of a custom array layout.
    """
    laysdict = {}

    arrayinfo, _ = resolve_layout(layout)
    allants = list(arrayinfo["antnames"])
    all_locations = list(arrayinfo["antlocations"])
    anant = len(all_locations)

    allsizes = arrayinfo["size"]
    allsizes = [allsizes] * anant if isinstance(allsizes, float | int) else list(allsizes)

    alltelnames = _per_antenna_telescope_names(arrayinfo, anant)

    if subarray_list:
        ant_to_idx = {name: i for i, name in enumerate(allants)}
        antnames = subarray_list
        antlocations = []
        antsizes = []
        anttelnames = []
        unknown = [ant for ant in antnames if ant not in ant_to_idx]
        if unknown:
            # A named subarray (meerkat, skamid-aa1, ...) is registered as a telescope in its
            # own right, so it belongs to --telescope, not here; users reach for -sublist first.
            raise ValueError(
                f"Unknown antenna(s) {', '.join(unknown)} for layout '{layout}'. "
                f"-sublist takes antenna names, e.g. {', '.join(allants[:3])}. "
                f"To use a named subarray, pass it as --telescope instead."
            )
        for ant in antnames:
            idx = ant_to_idx[ant]
            antlocations.append(all_locations[idx])
            antsizes.append(allsizes[idx])
            anttelnames.append(alltelnames[idx])

    elif subarray_range:
        if len(subarray_range) == 2:
            user_idx = list(range(subarray_range[0], subarray_range[1] + 1))
        elif len(subarray_range) == 3:
            user_idx = list(range(subarray_range[0], subarray_range[1], subarray_range[2]))
        else:
            raise ValueError(f"--subarray-range takes start,end or start,end,step; got {len(subarray_range)} value(s).")

        out_of_range = [i for i in user_idx if not 0 <= i < len(allants)]
        if out_of_range:
            # Report the span, not every index: a typo like 0,9999 otherwise prints thousands.
            raise ValueError(
                f"--subarray-range selects antenna indices {min(out_of_range)}-{max(out_of_range)} "
                f"outside layout '{layout}', which has {len(allants)} antennas "
                f"(valid indices 0-{len(allants) - 1})."
            )

        antnames = [allants[i] for i in user_idx]
        antlocations = [all_locations[i] for i in user_idx]
        antsizes = [allsizes[i] for i in user_idx]
        anttelnames = [alltelnames[i] for i in user_idx]

    elif subarray_file:
        subarray_data = load_yaml(subarray_file)

        if "antnames" in subarray_data:
            ant_to_idx = {name: i for i, name in enumerate(allants)}
            antnames = subarray_data["antnames"]
            antlocations = []
            antsizes = []
            anttelnames = []
            for ant in antnames:
                idx = ant_to_idx[ant]
                antlocations.append(all_locations[idx])
                antsizes.append(allsizes[idx])
                anttelnames.append(alltelnames[idx])

    laysdict = dict(
        centre=arrayinfo["centre"],
        antlocations=antlocations,
        antnames=antnames,
        size=antsizes,
        telescope_name=anttelnames,
        coord_sys=arrayinfo["coord_sys"],
        mount=arrayinfo["mount"],
        issubarray=True,
    )

    return laysdict


SIMMS_TELESCOPES = simms_telescopes()
