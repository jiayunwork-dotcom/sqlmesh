from __future__ import annotations

import typing as t
from collections import deque

from sqlglot import exp

from sqlmesh.utils.errors import ConfigError, SQLMeshError
from sqlmesh.utils.pydantic import PydanticModel

if t.TYPE_CHECKING:
    from sqlmesh.core.model import Model

# A join path is an ordered list of references, starting at the source model.
ReferencePath = t.List["Reference"]


class Reference(PydanticModel, frozen=True):
    model_name: str
    expression: exp.Expr
    unique: bool = False
    _name: str = ""

    @property
    def columns(self) -> t.List[str]:
        expression = self.expression
        if isinstance(expression, exp.Alias):
            expression = expression.this
        expression = expression.unnest()
        if isinstance(expression, (exp.Array, exp.Tuple)):
            return [e.output_name for e in expression.expressions]
        return [expression.output_name]

    @property
    def name(self) -> str:
        if not self._name:
            keys = []

            if isinstance(self.expression, (exp.Tuple, exp.Array)):
                for e in self.expression.expressions:
                    if not e.output_name:
                        raise ConfigError(
                            f"Reference '{e}' must have an inferrable name or explicit alias."
                        )
                    keys.append(e.output_name)
            elif self.expression.output_name:
                keys.append(self.expression.output_name)
            else:
                raise ConfigError(
                    f"Reference '{self.expression}' must have an inferrable name or explicit alias."
                )

            self._name = "__".join(keys)
        return self._name


class ReferenceGraph:
    def __init__(self, models: t.Iterable[Model]):
        self._model_refs: t.Dict[str, t.Dict[str, Reference]] = {}
        self._ref_models: t.Dict[str, t.Set[str]] = {}
        self._dim_models: t.Dict[str, t.Set[str]] = {}
        # Reference names that are a model's grain (unique targets), per model.
        self._grain_names: t.Dict[str, t.Set[str]] = {}

        for model in models:
            self.add_model(model)

    def add_model(self, model: Model) -> None:
        """Add a model and its references to the graph.

        Args:
            model: the model to add.
        """
        for column in model.columns_to_types or {}:
            self._dim_models.setdefault(column, set())
            self._dim_models[column].add(model.name)

        self._grain_names.setdefault(model.name, set())
        grains = getattr(model, "grains", None)
        if isinstance(grains, (list, tuple)):
            for grain in grains:
                self._grain_names[model.name].add(
                    Reference(model_name=model.name, expression=grain).name
                )
        else:
            # No grain metadata available (e.g. lightweight test doubles);
            # remember the absence explicitly so uniqueness falls back to the
            # flags declared on each reference.
            self._grain_names[model.name] = None

        for ref in model.all_references:
            self._model_refs.setdefault(model.name, {})
            self._model_refs[model.name][ref.name] = ref
            self._ref_models.setdefault(ref.name, set())
            self._ref_models[ref.name].add(model.name)

    def _is_unique_hop(self, from_model: str, to_model: str, ref_name: str) -> bool:
        """Whether joining ``from_model`` to ``to_model`` on ``ref_name`` is unique.

        A hop is unique when the target model's grain contains the shared
        reference name: each key of the source then matches at most one row of
        the target. Otherwise the target sits on the "many" side (e.g. all
        activities of a customer) and joining would fan the source out.

        When no grain metadata is registered for the target (e.g. test doubles
        that only expose ``all_references``), the reference's own ``unique``
        flag is used instead.
        """
        grains = self._grain_names.get(to_model)
        if grains is None:
            target_refs = self._model_refs.get(to_model, {})
            ref = target_refs.get(ref_name)
            return ref.unique if ref is not None else True
        return ref_name in grains
    def models_for_column(self, source: str, column: str, max_depth: int = 3) -> t.List[str]:
        """Find all the models with a column that join to a source within max_depth.

        Args:
            source: The source model.
            column: The column to look for.
            max_depth: The maximum number of models to join to find a path.

        Returns:
            The list of models that fit the criteria of the search.
        """
        models = []
        resolved_source = self._resolve_model_name(source) or source

        for model in self._dim_models[column]:
            try:
                if model != resolved_source:
                    self.find_path(resolved_source, model)
                models.append(model)
            except SQLMeshError:
                pass

        return sorted(models)

    def columns_of(self, model: str) -> t.Set[str]:
        """All columns known for a model (an empty set when the model is unknown).

        Accepts both the graph's unqualified model names and fully qualified
        names that include the default catalog.
        """
        names = self._model_refs.keys()
        resolved = self._resolve_model_name(model)
        if resolved is None:
            return set()
        return {
            column
            for column, models in self._dim_models.items()
            if resolved in models
        }

    def _resolve_model_name(self, model: str) -> t.Optional[str]:
        """Map a possibly catalog-qualified name to a graph model name."""
        if model in self._model_refs:
            return model
        parts = model.split(".")
        for length in range(len(parts), 0, -1):
            candidate = ".".join(parts[-length:])
            if candidate in self._model_refs:
                return candidate
        return None

    def _require_model(self, model: str) -> str:
        resolved = self._resolve_model_name(model)
        if resolved is None:
            raise SQLMeshError(f"Unknown model '{model}' referenced by a metric filter")
        return resolved

    def find_eligibility_path(
        self, source: str, target: str, max_depth: int = 3
    ) -> ReferencePath:
        """Find a path used for *eligibility* rather than for measure aggregation.

        Unlike :meth:`find_path`, the target model is allowed to sit on the "many"
        side of the relationship: a fact may be related to several rows of the
        target (e.g. all activities of a customer).  Such a non-unique hop is only
        permitted as the final hop of the path, because eligibility is evaluated
        per fact and must never multiply measures.

        Raises:
            SQLMeshError: when no path exists or when several equally short,
                equally unique paths reach the target and none can be uniquely
                selected.
        """
        if source == target:
            return []

        source = self._require_model(source)
        target = self._require_model(target)

        candidates = self._enumerate_paths(
            source,
            target,
            max_depth=max_depth,
            allow_terminal_many=True,
        )

        if not candidates:
            raise SQLMeshError(_no_path_message(source, target))

        if len(candidates) == 1:
            return candidates[0]

        # Prefer paths whose final hop is unique (the target behaves as a
        # unique dimension). Ambiguity remains only among truly equivalent
        # paths, in which case picking one silently would be unsafe.
        unique_endings = [path for path in candidates if self._path_ends_unique(path)]
        if len(unique_endings) == 1:
            return unique_endings[0]
        if len(unique_endings) > 1:
            candidates = unique_endings

        descriptions = " | ".join(
            " -> ".join([source] + [ref.model_name for ref in path]) for path in candidates
        )
        raise SQLMeshError(
            f"Cannot uniquely resolve the filter relationship from '{source}' to '{target}'. "
            f"Multiple equally short reference paths exist ({descriptions}). Add an explicit "
            "reference or disambiguate the models so a single path can be selected."
        )

    def _path_ends_unique(self, path: ReferencePath) -> bool:
        if len(path) < 2:
            return True
        a_ref, b_ref = path[-2], path[-1]
        return self._is_unique_hop(a_ref.model_name, b_ref.model_name, b_ref.name)

    def _enumerate_paths(
        self,
        source: str,
        target: str,
        max_depth: int,
        allow_terminal_many: bool,
    ) -> t.List[ReferencePath]:
        """Enumerate shortest simple paths from source to target.

        Edges are formed when two models share a reference name. When
        ``allow_terminal_many`` is False (regular measure joins) every hop must
        be unique; when True (eligibility) a non-unique hop is allowed only as
        the final hop.
        """
        if source not in self._model_refs:
            return []

        candidates: t.List[ReferencePath] = []
        best_length: t.Optional[int] = None

        # BFS. A path stores one reference per visited model, in visit order:
        # path[0] is one of the source's references, and every later reference
        # is declared by its own model. A model is reachable from the current
        # path when it shares a reference name with the path's last reference;
        # after that hop the walk continues through the reached model's other
        # references, which is how multi-hop paths change join keys.
        queue: t.Deque[ReferencePath] = deque(
            [ref] for ref in self._model_refs[source].values()
        )

        while queue:
            path = queue.popleft()
            visited = {ref.model_name for ref in path}
            last_ref = path[-1]
            current = last_ref.model_name

            for model_name in sorted(self._ref_models.get(last_ref.name, set())):
                if model_name in visited:
                    continue

                hop_ref = self._model_refs[model_name].get(last_ref.name)
                if hop_ref is None:
                    continue

                hop_unique = self._is_unique_hop(current, model_name, last_ref.name)

                if model_name == target:
                    if not allow_terminal_many and not hop_unique:
                        continue
                    new_path = path + [hop_ref]
                    if best_length is None:
                        best_length = len(new_path)
                    if len(new_path) == best_length:
                        candidates.append(new_path)
                    continue

                # The hop itself is represented by the reached model's
                # reference sharing the incoming key. Extend the walk through
                # that model's other references to open subsequent hops.
                if hop_unique:
                    for extension in self._model_refs[model_name].values():
                        if extension.name == hop_ref.name:
                            continue
                        new_path = path + [extension]
                        if len(new_path) < max_depth:
                            queue.append(new_path)

        return candidates

    def find_path(self, source: str, target: str, max_depth: int = 3) -> t.List[Reference]:
        """Find a path from source model to target model with max depth.

        Every hop must be unique: regular measure and grouping joins are never
        allowed to fan out through a many-to-many relationship.

        Args:
            source: The source model.
            target: The target model.
            max_depth: The maximum number of models to join to find a path.

        Returns:
            The list of references representing the join path of source to target.
        """
        source = self._require_model(source)
        target = self._require_model(target)
        candidates = self._enumerate_paths(
            source, target, max_depth=max_depth, allow_terminal_many=False
        )
        if candidates:
            return candidates[0]

        raise SQLMeshError(
            f"Cannot find path between '{source}' and '{target}'. Make sure that references/grains are configured and that a many to many join is not occurring."
        )


def _no_path_message(source: str, target: str) -> str:
    return (
        f"Cannot find path between '{source}' and '{target}'. Make sure that references/grains "
        "are configured and that a many to many join is not occurring."
    )
