import typing as t

from emevo.body import Body, Encount
from .core import Manager


def reproduce_and_update(
    manager: Manager,
    bodies: t.List[Body],
    encounts: t.List[Encount],
    **updates,
) -> None:
    if manager.is_asexual:
        for body in bodies:
            if manager.reproduce(body):
                manager.update_status(body, **updates)
    else:
        for encount in encounts:
            if manager.reproduce(encount):
                for body in encount.bodies:
                    manager.update_status(body, **updates)
