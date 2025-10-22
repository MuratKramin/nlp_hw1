# main.py
import json
import logging
import os
import re
from contextlib import asynccontextmanager
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import uvicorn
from fastapi import Depends, FastAPI, HTTPException, Request
from pydantic import BaseModel

# =======================
# Настройки и логирование
# =======================

KW_SUBP = r"(?:подпункт\w*|подпп\.|подп\.|пп\.)"

# Включить/выключить авто-самотесты при старте сервера
RUN_STARTUP_SELFTESTS = True

LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("law-links-service")

# ==========================
# Pydantic-модели ответа API
# ==========================

class LawLink(BaseModel):
    law_id: Optional[int] = None
    article: Optional[str] = None
    point_article: Optional[str] = None
    subpoint_article: Optional[str] = None


class LinksResponse(BaseModel):
    links: List[LawLink]


class TextRequest(BaseModel):
    text: str


# ======================
# Внутренние структуры
# ======================

@dataclass(frozen=True)
class ParsedRef:
    law_id: int
    article: Optional[str]
    point: Optional[str]
    subpoint: Optional[str]


# =================================
# Нормализация и вспомогательные ф-ии
# =================================

QUOTE_CHARS = {
    "«": '"', "»": '"',
    "“": '"', "”": '"',
    "„": '"', "‟": '"',
    "′": "'", "‚": '"',
}
DASH_CHARS = {"–": "-", "—": "-", "-": "-"}

L2C_MAP = str.maketrans({
    # латинские -> кириллица (look-alikes)
    "A": "А", "a": "а",
    "B": "В", "b": "в",
    "E": "Е", "e": "е",
    "K": "К", "k": "к",
    "M": "М", "m": "м",
    "H": "Н", "h": "н",
    "O": "О", "o": "о",
    "P": "Р", "p": "р",
    "C": "С", "c": "с",
    "T": "Т", "t": "т",
    "X": "Х", "x": "х",
    "Y": "У", "y": "у",
})

def normalize_text(s: str) -> str:
    """Нормализует кавычки/дефисы/пробелы (без латиница→кириллица!)."""
    for a, b in QUOTE_CHARS.items():
        s = s.replace(a, b)
    for a, b in DASH_CHARS.items():
        s = s.replace(a, b)
    s = re.sub(r"[ \t]{2,}", " ", s)
    return s
L2C_MAP = str.maketrans({
    # латинские → кириллица (look-alikes) — только для названий законов
    "A": "А", "a": "а",
    "B": "В", "b": "в",
    "E": "Е", "e": "е",
    "K": "К", "k": "к",
    "M": "М", "m": "м",
    "H": "Н", "h": "н",
    "O": "О", "o": "о",
    "P": "Р", "p": "р",
    "C": "С", "c": "с",
    "T": "Т", "t": "т",
    "X": "Х", "x": "х",
    "Y": "У", "y": "у",
})

def normalize_for_alias(s: str) -> str:
    """Нормализация строк именно для сопоставления названий законов."""
    s = normalize_text(s)
    s = s.translate(L2C_MAP) 
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s

def norm_alias_key(s: str) -> str:
    return normalize_for_alias(s)


def build_alias_maps(codex_aliases: Dict[str, List[str]]):
    """
    Возвращает:
      - alias_to_id: {нормализованный алиас -> law_id}
      - LAW_NAMED:    объединённый regex с именованными группами (?P<LID_15>...)
      - LAW_NONCAP:   тот же regex, но без именованных групп (?:...)
      - lid_group_names: {'LID_15': 15, ...}
    """
    alias_to_id: Dict[str, int] = {}
    lid_group_names: Dict[str, int] = {}
    parts: List[str] = []

    LOOKALIKE = {
        "А":"A","В":"B","Е":"E","К":"K","М":"M","Н":"H","О":"O","Р":"P","С":"C","Т":"T","Х":"X","У":"Y",
        "а":"a","в":"b","е":"e","к":"k","м":"m","н":"h","о":"o","р":"p","с":"c","т":"t","х":"x","у":"y",
    }

    def flex_char(ch: str) -> str:
        return f"[{re.escape(ch)}{LOOKALIKE[ch]}]" if ch in LOOKALIKE else re.escape(ch)

    UPPER_RU = r"[А-ЯЁ]"
    LOWER_RU = r"[а-яё]"
    RU_WORD   = r"[А-Яа-яЁё]+"

    def is_all_caps_short(tok: str) -> bool:
        # Аббревиатуры наподобие "НК", "УК", "ГПК", "КоАП" (последнее не all-caps, но короткое)
        return bool(re.fullmatch(rf"{UPPER_RU}{{2,5}}", tok))

    def flex_word(tok: str) -> str:
        """
        Гибкое слово:
          - 'РФ' обрабатываем посимвольно с LOOKALIKE
          - Короткие аббревиатуры (ГПК, НК...) без хвоста '[а-яё]*'
          - Для обычных слов — допускаем морф. окончания через '[а-яё]*'
          - Прилагательные на -ый/-ий/-ой 
        """
        if tok.upper() == "РФ":
            return "".join(flex_char(c) for c in tok)

        if is_all_caps_short(tok):
            return "".join(flex_char(c) for c in tok)

        if re.search(r"(ый|ий|ой)$", tok, flags=re.IGNORECASE):
            stem = tok[:-2]
            stem = "".join(flex_char(c) for c in stem)
            return fr"{stem}{LOWER_RU}+"

        stem = "".join(flex_char(c) for c in tok)
        return fr"{stem}{LOWER_RU}*"

    def alias_to_pattern(alias: str) -> str:
        alias = normalize_text(alias)
        tokens = re.split(r"(\s+)", alias)
        out = []
        for t in tokens:
            if t.isspace():
                out.append(r"\s+")
            elif re.fullmatch(RU_WORD, t) and len(t) >= 2:
                out.append(flex_word(t))
            else:
                out.append(re.escape(t))
        core = "".join(out)
        # Жёсткие границы слова вокруг ВЕСЬ алиаса, чтоб не матчить внутри слов
        return rf"(?<![0-9A-Za-zА-Яа-яЁё])(?:{core})(?![0-9A-Za-zА-Яа-яЁё])"

    # длинные алиасы — первыми
    all_items = []
    for lid_str, aliases in codex_aliases.items():
        lid = int(lid_str)
        for a in aliases:
            all_items.append((lid, a))
    all_items.sort(key=lambda x: len(x[1]), reverse=True)

    by_id: Dict[int, List[str]] = {}
    for lid, alias in all_items:
        alias_to_id[normalize_for_alias(alias)] = lid
        by_id.setdefault(lid, []).append(alias_to_pattern(alias))

    for lid, patts in by_id.items():
        gname = f"LID_{lid}"
        lid_group_names[gname] = lid
        parts.append(fr"(?P<{gname}>" + "|".join(patts) + ")")

    LAW_NAMED = "(?:" + "|".join(parts) + ")"
    LAW_NONCAP = re.sub(r"\(\?P<LID_\d+>", "(?:", LAW_NAMED)
    return alias_to_id, LAW_NAMED, LAW_NONCAP, lid_group_names



# ============
# Разбор значений
# ============

# Один "атом" значения: 3, 3.1, 3.4.5, буква (ru/en), допускаем кавычки вокруг
ATOM = r"(?:\d+(?:\.\d+)*|[«»\"'“”‘’]?[A-Za-zА-Яа-яё][»«\"'“”‘’]?)"

# Разделители перечисления: запятая, точка с запятой, "и", "или", "и/или", "либо", дефис для диапазона
VAL_CHUNK = fr"{ATOM}(?:\s*(?:,\s*|;\s*|и\s+|или\s+|и\/или\s+|либо\s+|-)\s*{ATOM})*"

def _is_letter(s: str) -> bool:
    return bool(re.fullmatch(r"[A-Za-zА-Яа-яё]", s))

def _expand_letter_range(a: str, b: str) -> List[str]:
    # Диапазон букв: поддерживаем кириллицу и латиницу
    a0, b0 = a.lower(), b.lower()
    if not (_is_letter(a0) and _is_letter(b0)):
        return [a, b]

    # Выберем алфавит по первой букве
    def is_lat(ch: str) -> bool:
        return bool(re.fullmatch(r"[a-z]", ch))

    def alphabet(ch: str) -> List[str]:
        if is_lat(ch):
            return [chr(c) for c in range(ord('a'), ord('z')+1)]
        else:
            # русская азбука без 'ё' последовательно; добавим 'ё' в нужное место
            # Для простоты возьмём непрерывный диапазон в Unicode от 'а' до 'я' и отдельно учтём 'ё'
            rus = [chr(c) for c in range(ord('а'), ord('я')+1)]
            # иногда встречается 'ё' — добавим её в алфавит (поставим после 'е')
            if 'ё' not in rus:
                rus.insert(rus.index('е')+1, 'ё')
            return rus

    if is_lat(a0) != is_lat(b0):
        # разные алфавиты — не расширяем
        return [a, b]

    alpha = alphabet(a0)
    try:
        ia, ib = alpha.index(a0), alpha.index(b0)
    except ValueError:
        return [a, b]

    if ib < ia:
        ia, ib = ib, ia
    return alpha[ia:ib+1]



def _expand_numeric_range(a: str, b: str) -> List[str]:
    # Поддержка:
    #  "1-3" -> 1,2,3
    #  "43.2-6" -> 43.2,43.3,43.4,43.5,43.6
    #  "1.1-1.3" -> 1.1,1.2,1.3
    def parse_levels(x: str) -> List[int]:
        return [int(t) for t in x.split(".")]

    def join_levels(levels: List[int]) -> str:
        return ".".join(str(t) for t in levels)

    if _is_letter(a) or _is_letter(b):
        return [a, b]

    a_levels = parse_levels(a)
    b_levels = parse_levels(b) if "." in b else None

    # Случай типа 43.2-6
    if b_levels is None and "." in a:
        prefix = a_levels[:-1]
        a_last = a_levels[-1]
        b_last = int(b)
        if b_last < a_last:
            a_last, b_last = b_last, a_last
        return [join_levels(prefix + [k]) for k in range(a_last, b_last + 1)]

    # Случай 1-3
    if b_levels is None and "." not in a:
        a0, b0 = int(a), int(b)
        if b0 < a0:
            a0, b0 = b0, a0
        return [str(k) for k in range(a0, b0 + 1)]

    # Случай 1.1-1.3 (или 3.4.1-3.4.5) – варьируем последнюю компоненту
    if b_levels is not None and len(a_levels) == len(b_levels) and a_levels[:-1] == b_levels[:-1]:
        start, end = a_levels[-1], b_levels[-1]
        if end < start:
            start, end = end, start
        prefix = a_levels[:-1]
        return [join_levels(prefix + [k]) for k in range(start, end + 1)]

    # Иначе оставляем как есть (без расширения)
    return [a, b]



def _split_by_commas_and_conj(s: str) -> List[str]:
    """
    Режем перечисление по , ; и союзам 'и/или', 'или', 'либо', 'и'.
    ВАЖНО: если chunk — это ОДНА буква (например, 'и'), считаем это значением,
    а не союзом.
    """
    s = s.strip()
    if not s:
        return []
    # одиночная буква — это значение, не союз
    if re.fullmatch(r"[A-Za-zА-Яа-яё]", s):
        return [s]
    # обычный случай — режем по разделителям/союзам
    parts = re.split(r"\s*(?:,|;|и\/или|либо|или|и)\s*", s, flags=re.IGNORECASE)
    return [p for p in parts if p]


def parse_values(chunk: Optional[str], *, expand_hyphens: bool = True) -> List[str]:
    """
    Разворачивает '1, 2 и 3', 'а-в', '43.2-6' в список значений.
    Если expand_hyphens=False (для статей), НЕ разворачиваем диапазон, оставляем
    строку как есть ('51.8-7').
    """
    if not chunk:
        return []
    parts = _split_by_commas_and_conj(chunk)
    out: List[str] = []
    for p in parts:
        p = p.strip()
        if "-" in p and expand_hyphens:
            a, b = [q.strip() for q in p.split("-", 1)]
            if _is_letter(a) and _is_letter(b):
                out.extend(_expand_letter_range(a, b))
            else:
                out.extend(_expand_numeric_range(a, b))
        else:
            out.append(p)

    # Убираем дубли, сохраняя порядок
    seen = set()
    uniq = []
    for v in out:
        if v not in seen:
            seen.add(v)
            uniq.append(v)
    return uniq

# ===========================
# Компиляция общих шаблонов
# ===========================

# Ключевые слова (учитываем падежи + защищаем короткие аббревиатуры от вхождений внутрь слов)
KW_ART  = r"(?:ст(?:атья|атьи|атье|атью|\.?)\w*)"
KW_PNT  = r"(?:пункт\w*|(?<![А-Яа-яё])п\.)"
KW_PART = r"(?:част\w*|(?<![А-Яа-яё])ч\.)"  
KW_SUBP = r"(?:подпункт\w*|подп\.|пп\.)"

def compile_patterns(LAW_NAMED: str, LAW_NONCAP: str) -> Dict[str, re.Pattern]:
    PRE = r"(?:\b(?:в|во|на|к|ко|по|об|обо|о|от|со|с|для)\b\s*)?"
    TOK_ART    = fr"(?:{PRE}{KW_ART})"
    TOK_PNPART = fr"(?:{PRE}(?:{KW_PNT}|{KW_PART}))"
    TOK_SUBP   = fr"(?:{PRE}{KW_SUBP})"

    LA_AFTER_SUBP  = fr"(?=(?:\s*[,:;]?\s*(?:{TOK_PNPART}|{TOK_ART}))|[),.;]|$)"
    LA_AFTER_POINT = fr"(?=(?:\s*[,:;]?\s*(?:{TOK_ART}|{TOK_SUBP}))|[),.;]|$)"
    # БЕЗ именованных групп:
    LA_AFTER_ART   = fr"(?=(?:\s*(?:{LAW_NONCAP})|\s*[,:;]?\s*(?:{TOK_PNPART}|{TOK_SUBP})|[),.;]|$))"

    patt_after = re.compile(
        fr"(?P<full>"
        fr"(?:{TOK_SUBP}\s*(?P<subp_vals>{VAL_CHUNK}?){LA_AFTER_SUBP}\s*[;,]?\s*)?"
        fr"(?:{TOK_PNPART}\s*(?P<point_vals>{VAL_CHUNK}?){LA_AFTER_POINT}\s*[;,]?\s*)?"
        fr"{TOK_ART}\s*(?P<article_vals>{VAL_CHUNK}?){LA_AFTER_ART}\s*"
        fr"(?P<law>{LAW_NAMED})"  
        fr")",
        flags=re.IGNORECASE
    )

    patt_before = re.compile(
        fr"(?P<full>"
        fr"(?P<law>{LAW_NAMED})\s*[,:;]?\s*"
        fr"{TOK_ART}\s*(?P<article_vals>{VAL_CHUNK}?){LA_AFTER_ART}"
        fr"(?:\s*[,:;]?\s*{TOK_PNPART}\s*(?P<point_vals>{VAL_CHUNK}?){LA_AFTER_POINT})?"
        fr"(?:\s*[,:;]?\s*{TOK_SUBP}\s*(?P<subp_vals>{VAL_CHUNK}?)(?=(?:\s*[),.;])|$))?"
        fr")",
        flags=re.IGNORECASE
    )

    patt_mid = re.compile(
        fr"(?P<full>"
        fr"{TOK_PNPART}\s*(?P<point_vals>{VAL_CHUNK}?){LA_AFTER_POINT}\s*[;,]?\s*"
        fr"{TOK_ART}\s*(?P<article_vals>{VAL_CHUNK}?){LA_AFTER_ART}\s*"
        fr"(?P<law>{LAW_NAMED})"
        fr")",
        flags=re.IGNORECASE
    )

    return {"after": patt_after, "before": patt_before, "mid": patt_mid}




def spans_overlap(a: Tuple[int, int], b: Tuple[int, int]) -> bool:
    # Перекрываются, если один не полностью "до" другого
    return not (a[1] <= b[0] or b[1] <= a[0])


def prune_less_specific(items: List[dict]) -> List[dict]:
    """
    Удаляет записи без subpoint, если в том же текстовом интервале
    (пересекающиеся спаны) уже есть записи с тем же (law_id, article, point)
    и НЕНУЛЕВЫМ subpoint.
    """
    # Группируем по (law_id, article, point)
    groups: Dict[Tuple[int, Optional[str], Optional[str]], List[dict]] = {}
    for it in items:
        key = (it["law_id"], it["article"], it["point"])
        groups.setdefault(key, []).append(it)

    to_drop = set()
    for key, arr in groups.items():
        with_sub = [x for x in arr if x["subpoint"] is not None]
        no_sub = [x for x in arr if x["subpoint"] is None]
        if not with_sub or not no_sub:
            continue
        # если есть перекрытие спанов между записями с subpoint и без — удаляем без subpoint
        for a in no_sub:
            for b in with_sub:
                if spans_overlap(a["span"], b["span"]):
                    to_drop.add(id(a))
                    break

    return [x for x in items if id(x) not in to_drop]



# ===========================
# Основной распознаватель
# ===========================

def extract_law_id_from_match(m: re.Match, lid_group_names: Dict[str, int]) -> Optional[int]:
    gd = m.groupdict()
    for name, lid in lid_group_names.items():
        if gd.get(name):
            return lid
    return None

from itertools import product
import re
from typing import Dict, List, Optional

_NUMERIC_POINT_RE = re.compile(r"^\d+(?:\.\d+)*$")
_SINGLE_LETTER_RE = re.compile(r"^[A-Za-zА-Яа-яё]$")

def _is_numeric_point(value: Optional[str]) -> bool:
    """Пункт/часть должны быть числовыми (разрешаем 3, 3.4, 10.1.2). None — ок."""
    if value is None:
        return True
    return bool(_NUMERIC_POINT_RE.fullmatch(value))

def detect_links(
    text: str,
    codex_aliases: Dict[str, List[str]],
) -> List["ParsedRef"]:
    """
    Извлекает юридические ссылки из текста.

    Ключевые решения:
      - Статьи НЕ разворачиваем по дефисам (например, '51.8-7' остаётся как есть),
        чтобы совпадать с эталонными тестами.
      - Пункты и подпункты разворачиваем (диапазоны/перечисления поддерживаются).
      - Буквенные пункты допустимы (например, 'п. с').
    """
    # 1) Нормализация и подготовка шаблонов
    text_norm = normalize_text(text)
    _, LAW_NAMED, LAW_NONCAP, lid_group_names = build_alias_maps(codex_aliases)
    pats = compile_patterns(LAW_NAMED, LAW_NONCAP)

    # 2) Собираем все совпадения из разных «порядков»
    matches: List[Tuple[int, int, re.Match]] = []
    for patt in pats.values():
        for m in patt.finditer(text_norm):
            matches.append((m.start(), m.end(), m))
    matches.sort(key=lambda t: (t[0], t[1]))

    # 3) Преобразуем совпадения в сырые элементы
    raw_items: List[Dict] = []
    for st, en, m in matches:
        law_id = extract_law_id_from_match(m, lid_group_names)
        if law_id is None:
            continue

        gd = m.groupdict()
        # ВАЖНО: статьи НЕ разворачиваем (expand_hyphens=False),
        # пункты/подпункты — разворачиваем.
        art_vals = parse_values(gd.get("article_vals"), expand_hyphens=False)
        pnt_vals = parse_values(gd.get("point_vals"),   expand_hyphens=True)
        sub_vals = parse_values(gd.get("subp_vals"),    expand_hyphens=True)

        arts = art_vals or [None]
        pnts = pnt_vals or [None]
        subs = sub_vals or [None]

        for a, p, s in product(arts, pnts, subs):
            raw_items.append({
                "law_id": law_id,
                "article": a,
                "point": p,
                "subpoint": s,
                "span": (st, en),
            })

    # 4) Удаляем менее специфичные записи (без подпункта рядом с теми же law/art/point)
    raw_items = prune_less_specific(raw_items)

    # 5) Дедупликация с сохранением порядка
    seen = set()
    result: List[ParsedRef] = []
    for it in raw_items:
        key = (it["law_id"], it["article"], it["point"], it["subpoint"])
        if key in seen:
            continue
        seen.add(key)
        result.append(ParsedRef(
            law_id=it["law_id"],
            article=it["article"],
            point=it["point"],
            subpoint=it["subpoint"],
        ))

    return result


# ============================
# FastAPI: lifecycle и эндпойнты
# ============================

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    try:
        law_path = Path(__file__).with_name("law_aliases.json")
        with law_path.open("r", encoding="utf-8") as f:
            codex_aliases = json.load(f)
        app.state.codex_aliases = codex_aliases
        logger.info("✅ Загружены алиасы законов: %d законов", len(codex_aliases))
    except Exception as e:
        logger.exception("Не удалось загрузить law_aliases.json: %s", e)
        raise

    # Автосамотесты (без HTTP), чтобы сразу увидеть, что парсер живой
    if RUN_STARTUP_SELFTESTS:
        try:
            logger.info("🧪 Запуск стартовых тестов...")
            _run_self_tests(codex_aliases)
            logger.info("🧪 Тесты завершены.")
        except Exception as e:
            logger.exception("Самотесты завершились с ошибкой: %s", e)

    yield

    # Shutdown
    try:
        del app.state.codex_aliases
    except Exception:
        pass
    logger.info("🛑 Сервис завершается...")


def get_codex_aliases(request: Request) -> Dict[str, List[str]]:
    return request.app.state.codex_aliases


app = FastAPI(
    title="Law Links Service",
    description="Сервис для выделения юридических ссылок из текста",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


@app.post("/detect", response_model=LinksResponse)
async def detect_endpoint(
    data: TextRequest,
    codex_aliases: Dict[str, List[str]] = Depends(get_codex_aliases),
):
    text = data.text or ""
    try:
        refs = detect_links(text, codex_aliases)
        # Логируем, но аккуратно (не спамим при больших текстах)
        logger.info("Обнаружено ссылок: %d", len(refs))
        # Подготовим ответ
        links = [
            LawLink(
                law_id=r.law_id,
                article=r.article if r.article is not None else None,
                point_article=r.point if r.point is not None else None,
                subpoint_article=r.subpoint if r.subpoint is not None else None,
            )
            for r in refs
        ]
        return LinksResponse(links=links)
    except Exception as e:
        logger.exception("Ошибка /detect: %s", e)
        # Возвращаем 500, чтобы явно показать сбой
        raise HTTPException(status_code=500, detail="Internal parsing error")


# ============================
# Тесты (локальные кейсы)
# ============================

def _run_self_tests(codex_aliases: Dict[str, List[str]]) -> None:
    """Мини-набор быстрых smoke-тестов парсера."""
    tests = [
        # Ожидается 3 ссылки с subpoint 1/2/3
        ("пп. 1, 2 и 3 п. 2 ст. 3 НК РФ", {"expect_count": 3,
                                           "expect_article": "3",
                                           "expect_point": "2",
                                           "expect_subpoints": {"1", "2", "3"}}),
        # Закон вначале
        ("УК РФ, ст. 145, п. 2, подп. б", {"expect_count": 1,
                                           "expect_article": "145",
                                           "expect_point": "2",
                                           "expect_subpoints": {"б"}}),
        # Десятичная статья
        ("ч. 3, ст. 30.1 КоАП РФ", {"expect_count": 1,
                                    "expect_article": "30.1"}),
        # Диапазон в статье с укороченной правой границей
        ("ст. 43.2-6 НК РФ", {"expect_count": 1,"expect_article": "43.2-6"}),
        # Подпункты с буквами и перечислением пунктов
        ("в подпунктах а, б и в пункта 3.345, 23 в статье 66 НК РФ",
         {"expect_min_count": 3, "expect_article": "66"}),
    ]

    for text, expectations in tests:
        refs = detect_links(text, codex_aliases)
        logger.debug("TEST: %s -> %s", text, refs)
        if "expect_count" in expectations:
            assert len(refs) == expectations["expect_count"], f"Ожидали {expectations['expect_count']}, получили {len(refs)}"
        if "expect_min_count" in expectations:
            assert len(refs) >= expectations["expect_min_count"], f"Ожидали минимум {expectations['expect_min_count']}, получили {len(refs)}"
        if "expect_article" in expectations and refs:
            assert any(r.article == expectations["expect_article"] for r in refs), "Статья не совпала"
        if "expect_point" in expectations and refs:
            assert any(r.point == expectations["expect_point"] for r in refs), "Пункт не совпал"
        if "expect_subpoints" in expectations and refs:
            got = {r.subpoint for r in refs if r.subpoint is not None}
            assert expectations["expect_subpoints"].issubset(got), f"Подпункты не совпали: {got}"
        if "expect_articles" in expectations and refs:
            got = {r.article for r in refs if r.article is not None}
            assert expectations["expect_articles"] == got, f"Статьи не совпали: {got}"

    logger.info("Самотесты пройдены: %d кейсов", len(tests))


# ============================
# Точка входа
# ============================

if __name__ == "__main__":
    logger.info("🚀 Сервис запускается...")
    uvicorn.run(app, host="0.0.0.0", port=8978)
