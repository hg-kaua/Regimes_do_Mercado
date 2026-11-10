"""Tratamento do aquecimento das janelas e divisão cronológica (Etapa 03). Funções puras."""

from __future__ import annotations

from math import floor

import pandas as pd

SPLITS = ("train", "validation", "test")


def drop_warmup(feat: pd.DataFrame, columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove as linhas iniciais em que algum indicador ainda não está definido.

    Só remove o bloco **inicial** (aquecimento das janelas). Qualquer NaN depois da
    primeira linha completa indica problema nos dados e gera ``ValueError``, em vez
    de ser descartado em silêncio.

    Returns:
        (tabela sem aquecimento, linhas removidas)
    """
    df = feat.sort_values("date").reset_index(drop=True)
    complete = df[columns].notna().all(axis=1)
    if not complete.any():
        raise ValueError("nenhuma linha com todos os indicadores definidos")
    first = int(complete.to_numpy().argmax())
    tail_nan = ~complete.iloc[first:]
    if tail_nan.any():
        dates = df.loc[tail_nan[tail_nan].index, "date"].dt.strftime("%Y-%m-%d").tolist()
        raise ValueError(f"valores ausentes após o aquecimento em {dates}")
    return df.iloc[first:].reset_index(drop=True), df.iloc[:first].reset_index(drop=True)


def split_sizes(n: int, train: float, validation: float, test: float) -> tuple[int, int, int]:
    """Tamanhos dos blocos: treino e validação arredondados para baixo; teste recebe o resto."""
    if abs(train + validation + test - 1.0) > 1e-9:
        raise ValueError("proporções devem somar 1")
    n_train = floor(n * train)
    n_val = floor(n * validation)
    n_test = n - n_train - n_val
    if min(n_train, n_val, n_test) <= 0:
        raise ValueError(f"amostra pequena demais para a divisão: n = {n}")
    return n_train, n_val, n_test


def chronological_split(df: pd.DataFrame, train: float = 0.70, validation: float = 0.15,
                        test: float = 0.15) -> pd.DataFrame:
    """Adiciona a coluna ``split`` (train/validation/test) em blocos contíguos por data.

    Não embaralha: o treino é o bloco mais antigo e o teste o mais recente.
    """
    out = df.sort_values("date").reset_index(drop=True).copy()
    if out["date"].duplicated().any():
        raise ValueError("datas duplicadas")
    n_train, n_val, n_test = split_sizes(len(out), train, validation, test)
    out["split"] = pd.Categorical(["train"] * n_train + ["validation"] * n_val + ["test"] * n_test,
                                  categories=list(SPLITS), ordered=True)
    return out


def split_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Datas inicial/final e tamanho de cada conjunto."""
    g = df.groupby("split", observed=True)["date"]
    s = pd.DataFrame({"start": g.min(), "end": g.max(), "n": g.size()})
    s["share"] = s["n"] / s["n"].sum()
    return s.reindex(list(SPLITS))


def validate_split(df: pd.DataFrame) -> None:
    """Verifica a integridade da divisão; levanta ``AssertionError`` se algo falhar.

    - datas estritamente crescentes e únicas;
    - três conjuntos não vazios, sem interseção de datas;
    - cada conjunto é um bloco contíguo e a ordem é treino < validação < teste;
    - nenhuma linha fora de um conjunto.
    """
    d = df["date"]
    assert d.is_monotonic_increasing and d.is_unique, "datas fora de ordem ou duplicadas"
    assert df["split"].notna().all(), "linha sem conjunto"
    sets = {s: set(df.loc[df["split"] == s, "date"]) for s in SPLITS}
    assert all(sets.values()), "conjunto vazio"
    assert not (sets["train"] & sets["validation"] or sets["train"] & sets["test"]
                or sets["validation"] & sets["test"]), "datas repetidas entre conjuntos"
    codes = df["split"].cat.codes.to_numpy()
    assert (codes[1:] >= codes[:-1]).all(), "conjuntos não contíguos ou fora de ordem"
    s = split_summary(df)
    assert s.loc["train", "end"] < s.loc["validation", "start"], "treino invade validação"
    assert s.loc["validation", "end"] < s.loc["test", "start"], "validação invade teste"
