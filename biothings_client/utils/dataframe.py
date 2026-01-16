def _normalize_dataframe_for_arrow(df):
    """
    Normalize list-like columns for type-consistency
    """
    import pandas

    list_types = (list, tuple)
    try:
        import numpy

        list_types = (list, tuple, numpy.ndarray)
    except Exception:
        pass

    def _is_list_value(value):
        return isinstance(value, list_types)

    def _is_nan(value):
        result = pandas.isna(value)
        return result if isinstance(result, bool) else False

    def _is_null(value):
        if value is None:
            return True
        if _is_list_value(value):
            return False
        return _is_nan(value)

    def _coerce_list_value(value):
        if value is None:
            return None
        if _is_list_value(value):
            return list(value)
        if _is_nan(value):
            return None
        return [value]

    def _list_entries_are_dicts(list_values):
        for value in list_values:
            for entry in value:
                if entry is None:
                    continue
                if not isinstance(entry, dict):
                    return False
        return True

    def _collect_key_stats(list_values):
        key_stats = {}
        for value in list_values:
            for entry in value:
                if entry is None:
                    continue
                for key, nested_value in entry.items():
                    stats = key_stats.setdefault(
                        key,
                        {"has_list": False, "has_non_list": False, "has_nan": False},
                    )
                    if _is_null(nested_value):
                        if nested_value is not None and _is_nan(nested_value):
                            stats["has_nan"] = True
                        continue
                    if _is_list_value(nested_value):
                        stats["has_list"] = True
                    else:
                        stats["has_non_list"] = True
        return key_stats

    def _normalize_list_entries(series, list_values):
        key_stats = _collect_key_stats(list_values)
        keys_with_list = {key for key, stats in key_stats.items() if stats["has_list"]}
        if not keys_with_list:
            return series

        keys_to_wrap = {
            key for key, stats in key_stats.items() if stats["has_list"] and stats["has_non_list"]
        }
        keys_with_nan = {key for key, stats in key_stats.items() if stats["has_list"] and stats["has_nan"]}

        def _normalize_entry(entry):
            if entry is None or not isinstance(entry, dict):
                return entry
            updated = None
            for key in keys_with_list:
                if key not in entry:
                    continue
                value = entry[key]
                if value is None:
                    continue
                if key in keys_with_nan and _is_nan(value):
                    if updated is None:
                        updated = dict(entry)
                    updated[key] = None
                    continue
                if key in keys_to_wrap and not _is_list_value(value) and not _is_nan(value):
                    if updated is None:
                        updated = dict(entry)
                    updated[key] = [value]
            return updated if updated is not None else entry

        def _normalize_value(value):
            if _is_null(value) or not _is_list_value(value):
                return value
            updated_list = None
            for idx, entry in enumerate(value):
                new_entry = _normalize_entry(entry)
                if new_entry is not entry:
                    if updated_list is None:
                        updated_list = list(value)
                    updated_list[idx] = new_entry
            return updated_list if updated_list is not None else value

        return series.map(_normalize_value)

    for column in df.columns:
        series = df[column]
        if series.dtype != "object":
            continue
        has_list = False
        has_non_list = False
        has_nan = False
        for value in series:
            if _is_null(value):
                if value is not None and _is_nan(value):
                    has_nan = True
                continue
            if _is_list_value(value):
                has_list = True
            else:
                has_non_list = True
            if has_list and has_non_list:
                break
        if has_list and (has_non_list or has_nan):
            series = series.map(_coerce_list_value)
            df[column] = series

        if not has_list:
            continue

        list_values = []
        for value in series:
            if _is_null(value):
                continue
            if not _is_list_value(value):
                list_values = []
                break
            list_values.append(value)
        if not list_values:
            continue

        if not _list_entries_are_dicts(list_values):
            continue

        df[column] = _normalize_list_entries(series, list_values)

    return df
