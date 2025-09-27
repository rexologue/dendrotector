"""Species configuration presets for dendrotector classifiers."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class SpeciesDefinition:
    """Represents a single tree or shrub species used by the classifier."""

    identifier: str
    common_name: str
    scientific_name: str
    aliases: Tuple[str, ...] = ()
    prompts: Tuple[str, ...] = ()

    def all_names(self) -> Tuple[str, ...]:
        """Return every name/alias associated with the species."""

        unique: list[str] = []
        seen: set[str] = set()
        for name in (self.common_name, *self.aliases, self.scientific_name):
            normalized = name.strip()
            if not normalized:
                continue
            lower = normalized.lower()
            if lower in seen:
                continue
            seen.add(lower)
            unique.append(normalized)
        return tuple(unique)


MOSCOW_REGION_SPECIES: Tuple[SpeciesDefinition, ...] = (
    SpeciesDefinition(
        identifier="silver_birch",
        common_name="Silver Birch",
        scientific_name="Betula pendula",
        aliases=("European white birch", "Warty birch"),
        prompts=(
            "street planting of silver birch trees in Moscow",
            "white bark of betula pendula tree",
        ),
    ),
    SpeciesDefinition(
        identifier="downy_birch",
        common_name="Downy Birch",
        scientific_name="Betula pubescens",
        aliases=("European downy birch",),
        prompts=("young downy birch near a river bank",),
    ),
    SpeciesDefinition(
        identifier="scots_pine",
        common_name="Scots Pine",
        scientific_name="Pinus sylvestris",
        aliases=("Scotch pine", "Common pine"),
        prompts=("tall pinus sylvestris pine in a temperate forest",),
    ),
    SpeciesDefinition(
        identifier="norway_spruce",
        common_name="Norway Spruce",
        scientific_name="Picea abies",
        aliases=("European spruce",),
        prompts=("dense picea abies spruce tree in winter",),
    ),
    SpeciesDefinition(
        identifier="siberian_spruce",
        common_name="Siberian Spruce",
        scientific_name="Picea obovata",
        aliases=("Obovate spruce", "Siberian spruce"),
        prompts=("picea obovata spruce growing in a russian city park",),
    ),
    SpeciesDefinition(
        identifier="english_oak",
        common_name="English Oak",
        scientific_name="Quercus robur",
        aliases=("Pedunculate oak", "Common oak"),
        prompts=("broad crown of quercus robur tree in summer",),
    ),
    SpeciesDefinition(
        identifier="norway_maple",
        common_name="Norway Maple",
        scientific_name="Acer platanoides",
        aliases=("European maple", "Platanoide maple"),
        prompts=("acer platanoides maple with dense summer foliage",),
    ),
    SpeciesDefinition(
        identifier="tatar_maple",
        common_name="Tatar Maple",
        scientific_name="Acer tataricum",
        aliases=("Tatarian maple", "Acer ginnala"),
        prompts=("ornamental acer tataricum shrub in a courtyard",),
    ),
    SpeciesDefinition(
        identifier="small_leaved_linden",
        common_name="Small-leaved Linden",
        scientific_name="Tilia cordata",
        aliases=("Littleleaf linden", "Lime tree"),
        prompts=("tilia cordata linden lining a european boulevard",),
    ),
    SpeciesDefinition(
        identifier="black_alder",
        common_name="Black Alder",
        scientific_name="Alnus glutinosa",
        aliases=("European alder",),
        prompts=("alnus glutinosa alder near wetland park",),
    ),
    SpeciesDefinition(
        identifier="european_rowan",
        common_name="European Rowan",
        scientific_name="Sorbus aucuparia",
        aliases=("Mountain ash", "Rowan tree"),
        prompts=("sorbus aucuparia with orange berries in autumn",),
    ),
    SpeciesDefinition(
        identifier="goat_willow",
        common_name="Goat Willow",
        scientific_name="Salix caprea",
        aliases=("Pussy willow",),
        prompts=("salix caprea goat willow shrub with catkins",),
    ),
    SpeciesDefinition(
        identifier="european_ash",
        common_name="European Ash",
        scientific_name="Fraxinus excelsior",
        aliases=("Common ash",),
        prompts=("fraxinus excelsior ash tree with compound leaves",),
    ),
    SpeciesDefinition(
        identifier="bird_cherry",
        common_name="Bird Cherry",
        scientific_name="Prunus padus",
        aliases=("Hackberry", "Mayday tree"),
        prompts=("prunus padus bird cherry blooming in spring",),
    ),
)

__all__ = ["SpeciesDefinition", "MOSCOW_REGION_SPECIES"]
