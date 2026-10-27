# Source dictionaries and explicit analysis cohorts

Run `PYTHONPATH=src .venv/bin/python docs/examples/metadata_dictionary.py` from the repository root. The standalone [synthetic example](metadata_dictionary.py) exports PNG, PDF and SVG under `.venv/dictionary-api/` using public dictionaries only, with no axes edits.

## Visual styles (options namespace)

```python
options = {
    "compact_layout": True,
    "metadata_legend_position": "bottom",
    "metadata_styles": {
        "treatment": {"legend": {"ncol": 2, "alignment": "top"},
                      "labels": {"A": "Treatment A"}},
        "response": {"legend": {"ncol": 1}}
    }
}
```

`metadata_styles` defaults to `{}`. Keys are original displayed metadata source columns, never prettified titles or `metadata:` keys. Entries accept only `legend` and `labels`. Legend accepts positive integer `ncol`/`nrow` and `alignment` (`top`/`center`). Each nested setting overrides the corresponding legacy scalar or source-mapping setting; unspecified settings retain their legacy fallback. When both row and column counts are supplied, columns are the maximum of ncol and the columns needed to fit the row limit. Counts do nothing for numeric colorbars; alignment still applies. Legend layout requires compact Matplotlib; unsupported combinations raise even if legends are hidden.

`labels` maps original stringified category values to display strings and works in both backends, including noncompact Matplotlib. Source labels override global `legend_label_overrides`, which remains the fallback; existing global keys are not reinterpreted. Display text is truncated using the existing max-character option. Identical values in different sources can have different labels; identical display labels do not merge categories. Source data, colors, category order and hover payloads remain unchanged. Missing markers retain their separate existing controls.

Legacy key namespaces are unchanged: legend offsets use keys such as `metadata:treatment`; source mappings/order/scales/heights use `treatment`; global labels use category strings; global titles use original title strings. `metadata_legend_order=None` preserves categorical-first/right-anchored numeric placement, whereas `[]` explicitly preserves input track order with sequential mixed packing. Neither hides legends.

## Population filters (top-level namespace)

```python
params = {
    "metadata_filter_mode": "analysis_cohort",
    "metadata_filters": {
        "treatment": {"include": ["A", "B"], "exclude": ["B"], "missing": "exclude"},
        "age": {"min": 18, "max": 65, "missing": "error"}
    }
}
```

Both parameters default to `None`. Nonempty filters require explicit `analysis_cohort`; display-only and unknown modes are rejected. `None`/`{}` do not filter. Each original source accepts only include/exclude sequences of non-null scalars, inclusive finite numeric min/max, and missing (`exclude` default, `include`, `error`). Predicates are ANDed within and across sources. Numeric ranges require numeric dtype, never coercing categorical strings. Use missing policy rather than null list entries. Empty include matches no observed values; empty exclude excludes no observed values; missing policy still applies. Empty source rules apply only their default missing policy. Unknown sources/operators and reversed bounds fail.

Missing includes samples absent from the metadata table. `include` bypasses only that source predicates, not other sources. Duplicate metadata sample identifiers are rejected, including identical records and missing-versus-nonmissing conflicts; collapse duplicates explicitly if appropriate. Metadata can be derived from mutation rows using existing `metadata_cols`; every derived sample must have consistent metadata. Filters do not modify input objects.

Filtering occurs after existing mutation-row and sample filters, before ranking, event collapse, counts, TMB and denominator preparation. Missing validation uses the eligible pre-ranking population: metadata removed by metadata_require_mutations and custom-TMB-only samples excluded by existing filter rules do not trigger errors. Retained metadata-only and cell-state samples still obey the missing policy. Cell states, custom TMB population and explicit sample order are restricted consistently. Filters never introduce metadata-only samples: existing `metadata_require_mutations`, `show_all_samples`, selected genes and cell-state inclusion rules continue to determine display population. Zero eligible samples raises a clear error.

**Scientific scope:** automatic `total_samples` choices (`any_mutations`, `all`, `oncoplot`) keep their existing definitions but operate on the filtered cohort. This argument is a mode, not an explicit numeric reference denominator. Explicit `gene_denominators` remain unchanged; external `row_summaries` counts/denominators remain unchanged (selected rows are still selected as usual). Such summaries describe an external reference population and need not describe the filtered tiles. No display-only filtering is offered because hiding samples without clarifying counts would misrepresent the analysis.

## Ownership and backend validation

Options instances and mappings are defensively copied and validated on every call, including nested mutations made after construction. Arbitrary Mapping containers (including nested read-only mapping proxies) are materialized as independent dictionaries. Options subclasses are accepted through their base fields: subclass-only attributes remain untouched on the caller and are not passed into the internal base options object. Direct API offsets never modify caller options. Scalar geometry must be finite; existing valid signed angles/nudges are retained. Compact incompatible options are rejected before figure allocation; render failures close only the owned Matplotlib figure. Caller-owned figures remain open.
