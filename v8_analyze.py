#!/usr/bin/env python3
"""
v8_analyze.py — Финальный психолингвистический портрет по VK-архиву (v8/).

Читает HTML-выгрузку ВКонтакте, извлекает тексты с нескольких позиций
каждого файла, проводит тематический / эмоциональный / культурный анализ
и отправляет всё AI-агенту (Ollama) для финализации психологического портрета.

Запуск:
    python v8_analyze.py                     # полный анализ + AI
    python v8_analyze.py --prompt-only       # только показать промпт
    python v8_analyze.py --save              # сохранить отчёт в logs/v8/
    python v8_analyze.py --model qwen2.5:14b
"""

import re
import sys
import time
import argparse
from pathlib import Path
from collections import Counter, defaultdict

from bs4 import BeautifulSoup
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich import box
from ollama import Client

# ── Конфигурация ───────────────────────────────────────────────────────────────
try:
    from config import selected_model, temperature, num_ctx
except ImportError:
    selected_model = "mistral-nemo:latest"
    temperature    = 0.5
    num_ctx        = 12000

V8_DIR = Path(__file__).parent / "v8"

# Байт читаем за один «чанк» из файла (читаем 3 чанка с разных позиций)
CHUNK_SIZE   = 180_000
# Максимум символов текста в финальном AI-промпте
AI_TEXT_CAP  = 18_000

SECTIONS_META = {
    "profile.html":      ("👤 Профиль",     "personal"),
    "wall.html":         ("📝 Стена",        "expression"),
    "comments.html":     ("💬 Комментарии",  "expression"),
    "messages.html":     ("✉️  Сообщения",    "private"),
    "audio.html":        ("🎵 Аудио",        "culture"),
    "video.html":        ("🎬 Видео",        "culture"),
    "likes.html":        ("❤️  Лайки",        "culture"),
    "bookmarks.html":    ("🔖 Закладки",     "interest"),
    "photos.html":       ("📷 Фото",         "personal"),
    "ads.html":          ("📢 Реклама",      "misc"),
    "apps.html":         ("📱 Приложения",   "misc"),
    "payments.html":     ("💳 Платежи",      "behavior"),
    "other.html":        ("📂 Прочее",       "misc"),
    "verification.html": ("✅ Верификация",  "misc"),
}

STOPWORDS = {
    "и","в","не","на","что","с","я","это","а","то","все","как","но","от","у","к",
    "по","он","она","они","же","бы","так","из","за","уже","о","мне","его","её",
    "их","мы","вы","был","было","быть","вот","для","если","ещё","нет","там",
    "тут","ну","да","со","во","при","до","об","или","тоже","только","когда",
    "тебя","меня","себя","может","этот","эта","этих","своего","свой","своя",
    "есть","нас","вас","им","ним","все","всё","всего","будет","этого","тебе",
    "себе","нам","вам","моя","мой","мои","the","and","to","of","is","in","it",
    "was","for","on","are","as","with","his","they","at","be","this","have",
    "from","or","had","by","not","but","what","all","were","we","when","your",
    "can","she","do","how","said","there","an","each","which","use","com",
    "http","https","html","www","jpg","png","gif","mp3","mp4","pdf","photo",
    "video","story","audio","image","attachment","deleted","none","просмотров",
    "просмотра","фотокарточка","карточка","гостинец","кинолента","запись",
    "ссылка","link","post","item","wall","feat","remix","original","extended",
    # Сокращения месяцев из дат VK
    "янв","фев","мар","апр","май","мая","июн","июл","авг","сен","окт","ноя","дек",
    "jan","feb","mar","apr","jun","jul","aug","sep","oct","nov","dec",
}

# ── Тематические кластеры ──────────────────────────────────────────────────────
THEMES: dict[str, list[str]] = {
    "paranoia_ti": [
        "пси","псих","машина","операторы","нлп","nlp","targeted","individual",
        "эмболия","пидор","ублюдки","подставляют","вещают","перехват","пытки",
        "шавки","агент","слежка","технологии","засекреченное","кодирование",
        "психушка","психущкина",
    ],
    "politics": [
        "путин","кремль","kremlin","россия","россиян","страна","война","санкции",
        "власть","государство","уеду","уехать","эмиграция","дагестанских","русский",
        "правительство","фсб","спецслужбы","секретский","оппозиция",
    ],
    "tech_hacker": [
        "python","dll","win32","tor","http","linux","kali","код","программа",
        "скрипт","алгоритм","сервер","уязвимость","хакер","эксплойт","socket",
        "github","bash","debug","reverse","exploit","binary","assembly","asm",
    ],
    "music": [
        "rave","neuropunk","dnb","hardcore","industrial","techno","drum","bass",
        "kmfdm","remix","bass","track","музыка","трек","лейбл","релиз","bpm",
        "mix","dj","soundcloud","bandcamp","рейв","микс","сет","club",
    ],
    "social_aggression": [
        "пиздец","ебать","нахуй","блять","сука","хуй","ублюдок","мудак","пидор",
        "дебил","идиот","быдло","тупой","урод","крыса","петух","придурок",
    ],
    "self_expression": [
        "думаю","считаю","знаю","понял","вижу","чувствую","хочу","мечтаю",
        "проблема","решение","идея","концепция","теория","исследование","открыл",
    ],
}

EMOTION_POSITIVE = {
    "хорошо","отлично","круто","молодец","классно","здорово","нравится","люблю",
    "рад","радость","спасибо","удачно","норм","nice","cool","love","good",
}
EMOTION_NEGATIVE = {
    "плохо","ужасно","страшно","боюсь","ненавижу","злость","грусть","горе",
    "безнадёжно","депрессия","тоска","страдание","боль","hate","bad","horrible",
}
EMOTION_AGGRESSION = {
    "пиздец","ебать","нахуй","блять","сука","хуй","бля","нахрен","ублюдок",
    "мудак","пидор","дебил","урод","задрот","быдло","тупой","крыса","петух",
}

console = Console()
client  = Client(host="127.0.0.1")


# ── Утилиты ────────────────────────────────────────────────────────────────────

def ts() -> str:
    return f"[dim]{time.strftime('%H:%M:%S')}[/dim]"


def clean(text: str) -> str:
    text = re.sub(r"https?://\S+", "", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    text = re.sub(r"https?://\S+", "", text)
    return [w.lower() for w in re.findall(r"[а-яёА-ЯЁa-zA-Z]{3,}", text)
            if w.lower() not in STOPWORDS]


def extract_domains(html_text: str) -> list[str]:
    domains = []
    for href in re.findall(r'href=["\']?(https?://[^\s"\'<>]+)', html_text):
        if "vk.com" in href or "userapi.com" in href:
            continue
        m = re.match(r"https?://([^/]+)", href)
        if m:
            domains.append(m.group(1).replace("www.", ""))
    return domains


# ── Парсинг ────────────────────────────────────────────────────────────────────

def read_chunks(filepath: Path) -> list[str]:
    """Читает файл тремя чанками: начало, середина, конец."""
    size = filepath.stat().st_size
    positions = [0]
    if size > CHUNK_SIZE * 2:
        positions.append(size // 2 - CHUNK_SIZE // 2)
    if size > CHUNK_SIZE * 3:
        positions.append(max(0, size - CHUNK_SIZE))

    chunks = []
    with open(filepath, "rb") as fh:
        for pos in positions:
            fh.seek(pos)
            raw = fh.read(CHUNK_SIZE)
            chunks.append(raw.decode("utf-8", errors="replace"))
    return chunks


def parse_section(filepath: Path) -> dict:
    """Парсит HTML-секцию (3 позиции файла), возвращает агрегированные данные."""
    stat = filepath.stat()
    result: dict = {
        "name":         filepath.name,
        "size_mb":      stat.st_size / (1024 * 1024),
        "texts":        [],
        "dates":        [],
        "domains":      [],
        "words":        Counter(),
        "items_parsed": 0,
        "themes":       Counter(),
        "emotions":     {"pos": 0, "neg": 0, "agg": 0},
    }

    chunks = read_chunks(filepath)
    seen_texts: set[str] = set()

    for chunk in chunks:
        soup = BeautifulSoup(chunk, "html.parser")

        for item in soup.select(".item"):
            mains = item.select(".item__main")
            raw_text = " ".join(d.get_text(" ") for d in mains)
            text = clean(raw_text)

            # Дедупликация
            key = text[:80]
            if key in seen_texts or len(text) < 8:
                continue
            seen_texts.add(key)

            result["items_parsed"] += 1
            result["texts"].append(text[:500])

            # Даты
            for d in item.select(".item__tertiary"):
                dt = clean(d.get_text())
                if dt:
                    result["dates"].append(dt[:60])

        # Слова
        all_text = " ".join(result["texts"])
        result["words"] = Counter(tokenize(all_text))

        # Домены
        result["domains"].extend(extract_domains(chunk))

    # Тематика и эмоции по всему тексту
    combined_words = set(tokenize(" ".join(result["texts"])))
    for theme, keywords in THEMES.items():
        for kw in keywords:
            if kw in combined_words:
                result["themes"][theme] += 1

    for w in tokenize(" ".join(result["texts"])):
        if w in EMOTION_POSITIVE:
            result["emotions"]["pos"] += 1
        if w in EMOTION_NEGATIVE:
            result["emotions"]["neg"] += 1
        if w in EMOTION_AGGRESSION:
            result["emotions"]["agg"] += 1

    return result


# ── Аналитика ──────────────────────────────────────────────────────────────────

def theme_summary(sections: dict[str, dict]) -> dict[str, int]:
    total: Counter = Counter()
    for d in sections.values():
        total.update(d["themes"])
    return dict(total.most_common())


def emotion_summary(sections: dict[str, dict]) -> dict:
    pos = sum(d["emotions"]["pos"] for d in sections.values())
    neg = sum(d["emotions"]["neg"] for d in sections.values())
    agg = sum(d["emotions"]["agg"] for d in sections.values())
    total = max(pos + neg + agg, 1)
    return {"pos": pos, "neg": neg, "agg": agg,
            "pct_pos": round(pos/total*100), "pct_neg": round(neg/total*100),
            "pct_agg": round(agg/total*100)}


def top_domains(sections: dict[str, dict], n: int = 30) -> list[tuple[str, int]]:
    c: Counter = Counter()
    for d in sections.values():
        c.update(d["domains"])
    return c.most_common(n)


def combined_words(sections: dict[str, dict], n: int = 80) -> list[tuple[str, int]]:
    c: Counter = Counter()
    for d in sections.values():
        c.update(d["words"])
    return c.most_common(n)


def timeline_hints(sections: dict[str, dict]) -> str:
    """Извлекает диапазон дат из всех секций."""
    years: Counter = Counter()
    for d in sections.values():
        for date_str in d["dates"]:
            m = re.search(r"\b(20\d{2})\b", date_str)
            if m:
                years[m.group(1)] += 1
    if not years:
        return "неизвестно"
    sorted_years = sorted(years.keys())
    return f"{sorted_years[0]} – {sorted_years[-1]}  (пик: {years.most_common(1)[0][0]})"


def collect_text_samples(sections: dict[str, dict]) -> str:
    """Собирает тексты по приоритету секций для AI-промпта."""
    priority = [
        "wall.html", "comments.html", "messages.html",
        "bookmarks.html", "audio.html", "video.html", "profile.html",
    ]
    budget = AI_TEXT_CAP
    parts  = []

    for fname in priority + [f for f in sections if f not in priority]:
        if fname not in sections or budget <= 0:
            continue
        label = SECTIONS_META.get(fname, (fname, ""))[0]
        # Берём разнообразные тексты — не только первые
        texts = sections[fname]["texts"]
        # каждый 3-й текст для разнообразия + первые 10
        sample_texts = texts[:10] + texts[10::3]
        block = "\n".join(t for t in sample_texts if len(t) > 15)[:budget]
        if block:
            parts.append(f"\n──── {label} ────\n{block}")
            budget -= len(block)

    return "\n".join(parts)


# ── Построение промпта ─────────────────────────────────────────────────────────

def compose_portrait_prompt(sections: dict[str, dict]) -> str:
    themes   = theme_summary(sections)
    emotions = emotion_summary(sections)
    domains  = top_domains(sections, 25)
    words    = combined_words(sections, 60)
    timeline = timeline_hints(sections)
    texts    = collect_text_samples(sections)

    # Суммарные числа
    total_items = sum(d["items_parsed"] for d in sections.values())
    total_size  = sum(d["size_mb"]      for d in sections.values())

    wall_n  = sections.get("wall.html",     {}).get("items_parsed", 0)
    msg_n   = sections.get("messages.html", {}).get("items_parsed", 0)
    comm_n  = sections.get("comments.html", {}).get("items_parsed", 0)
    audio_n = sections.get("audio.html",    {}).get("items_parsed", 0)
    video_n = sections.get("video.html",    {}).get("items_parsed", 0)
    like_n  = sections.get("likes.html",    {}).get("items_parsed", 0)

    # Тематические сигналы
    theme_lines = "\n".join(
        f"  {t}: {v} сигналов" for t, v in sorted(themes.items(), key=lambda x: -x[1])
    )
    domain_lines = "  " + ", ".join(f"{d}({n})" for d, n in domains[:20])
    word_lines   = "  " + ", ".join(f"{w}({n})" for w, n in words[:40])

    prompt = f"""Ты — ведущий эксперт в области цифровой форензики личности, психолингвистики
и культурологии. Твоя задача — составить окончательный, максимально конкретный
психологический портрет реального человека по его полной выгрузке данных ВКонтакте.

════════════════════════════════════════
  РАЗДЕЛ 1 — МЕТАДАННЫЕ АРХИВА
════════════════════════════════════════
Архив создан: 2026-03-30 (дата генерации index.html)
Временной диапазон активности: {timeline}
Общий объём: {total_size:.1f} MB
Записей на стене: {wall_n}   Сообщений: {msg_n}   Комментариев: {comm_n}
Аудио: {audio_n}   Видео: {video_n}   Лайков: {like_n}
Всего обработано записей (сэмпл): {total_items}

════════════════════════════════════════
  РАЗДЕЛ 2 — ТЕМАТИЧЕСКИЕ КЛАСТЕРЫ
════════════════════════════════════════
(автоматическое обнаружение ключевых слов)
{theme_lines}

════════════════════════════════════════
  РАЗДЕЛ 3 — ЭМОЦИОНАЛЬНЫЙ ПРОФИЛЬ
════════════════════════════════════════
Позитивные маркеры: {emotions['pos']} ({emotions['pct_pos']}%)
Негативные маркеры: {emotions['neg']} ({emotions['pct_neg']}%)
Агрессивные маркеры: {emotions['agg']} ({emotions['pct_agg']}%)

════════════════════════════════════════
  РАЗДЕЛ 4 — КУЛЬТУРНЫЕ МАРКЕРЫ
════════════════════════════════════════
Топ-20 внешних доменов (исключая vk.com):
{domain_lines}

Топ-40 значимых слов архива:
{word_lines}

════════════════════════════════════════
  РАЗДЕЛ 5 — РЕАЛЬНЫЕ ТЕКСТЫ ЧЕЛОВЕКА
════════════════════════════════════════
{texts}

════════════════════════════════════════
  ЗАДАНИЕ — ФИНАЛЬНЫЙ ПОРТРЕТ
════════════════════════════════════════

На основе ВСЕХ приведённых данных составь ПОЛНЫЙ и ОКОНЧАТЕЛЬНЫЙ психологический
портрет этого человека. Будь конкретным, аналитичным, избегай общих фраз.

## I. КТО ЭТОТ ЧЕЛОВЕК
- Примерный возраст, социальный контекст, место жизни (по косвенным признакам)
- Образование и интеллектуальный уровень
- Профессиональная/техническая идентичность

## II. ЛИНГВИСТИЧЕСКИЙ ПРОФИЛЬ
- Доминирующий языковой регистр (мат, сленг, техножаргон, интернет-язык)
- Характерные речевые паттерны и идиосинкразии
- Грамотность, структура мысли, связность текстов
- Языки и субъязыки (рус/eng/микс/нетяз)

## III. КУЛЬТУРНЫЙ КОД
- Музыкальные жанры и сцены (с конкретными названиями)
- Информационный рацион: что читает, смотрит, откуда берёт инфу
- Субкультурная идентификация
- Отношение к России / Западу / технологиям

## IV. ПСИХОЛОГИЧЕСКИЙ АНАЛИЗ
- Доминирующие психологические паттерны (тревожность, паранойя, агрессия, нарциссизм и т.д.)
- Механизмы защиты и самопрезентации
- Характер межличностных отношений (по переписке и комментариям)
- Признаки психического неблагополучия (если есть) — конкретно и без диагнозов
- Базовые страхи и потребности, которые просматриваются в текстах

## V. ЧТО ЧЕЛОВЕК ХОТЕЛ ДОНЕСТИ МИРУ
- Главные послания и нарративы в публичных постах
- Мотивация публикаций
- Аудитория, к которой обращается
- Нереализованные желания и амбиции, прочитываемые из текстов

## VI. ЖИЗНЕННАЯ ТРАЕКТОРИЯ
- Ключевые периоды активности по годам
- Как менялся стиль и темы
- Переломные моменты (если прослеживаются)

## VII. ФИНАЛЬНЫЙ ВЕРДИКТ
Одним абзацем (4–6 предложений): кто этот человек, что с ним происходит,
что он ищет, чего боится и что хочет донести — интегрированная характеристика.
"""
    return prompt


# ── Rich-вывод ─────────────────────────────────────────────────────────────────

def print_header():
    console.print(Panel(
        "[bold cyan]VK Archive — Psychological Portrait Engine[/bold cyan]\n"
        "[dim]Мультичанковый парсинг · Тематический кластеринг · "
        "Эмоциональный анализ · AI-психопортрет[/dim]",
        border_style="blue",
    ))


def print_stats_table(sections: dict[str, dict]):
    t = Table(
        box=box.ROUNDED,
        title="[bold cyan]📊 Статистика архива[/bold cyan]",
        header_style="bold blue",
    )
    t.add_column("Секция",     style="cyan",    width=20)
    t.add_column("Размер",     style="yellow",  justify="right", width=9)
    t.add_column("Записей",    style="green",   justify="right", width=8)
    t.add_column("Слов",       style="white",   justify="right", width=7)
    t.add_column("Доменов",    style="magenta", justify="right", width=8)

    for fname, data in sections.items():
        label = SECTIONS_META.get(fname, (fname, ""))[0]
        sz    = f"{data['size_mb']:.1f}M" if data["size_mb"] >= 1 else f"{data['size_mb']*1024:.0f}K"
        t.add_row(label, sz, str(data["items_parsed"]),
                  str(sum(data["words"].values())), str(len(data["domains"])))
    console.print(t)


def print_theme_table(sections: dict[str, dict]):
    themes = theme_summary(sections)
    emo    = emotion_summary(sections)

    th_table = Table(box=box.SIMPLE, title="[bold yellow]🧩 Тематические кластеры[/bold yellow]",
                     header_style="bold yellow")
    th_table.add_column("Кластер",   style="yellow")
    th_table.add_column("Сигналов",  justify="right", style="cyan")

    THEME_LABELS = {
        "paranoia_ti":      "🔴 Паранойя / TI",
        "politics":         "🏛  Политика",
        "tech_hacker":      "💻 Технологии / хак",
        "music":            "🎵 Музыка",
        "social_aggression":"💢 Агрессия",
        "self_expression":  "🗣  Самовыражение",
    }
    for key, cnt in sorted(themes.items(), key=lambda x: -x[1]):
        th_table.add_row(THEME_LABELS.get(key, key), str(cnt))

    em_table = Table(box=box.SIMPLE, title="[bold magenta]💡 Эмоциональный профиль[/bold magenta]",
                     header_style="bold magenta")
    em_table.add_column("Тип",        style="white")
    em_table.add_column("Маркеров",   justify="right", style="magenta")
    em_table.add_column("%",          justify="right", style="dim")
    em_table.add_row("[green]Позитив[/green]",    str(emo["pos"]), f"{emo['pct_pos']}%")
    em_table.add_row("[red]Негатив[/red]",        str(emo["neg"]), f"{emo['pct_neg']}%")
    em_table.add_row("[bold red]Агрессия[/bold red]", str(emo["agg"]), f"{emo['pct_agg']}%")

    console.print(Columns([th_table, em_table]))


def print_word_domain_tables(sections: dict[str, dict]):
    words   = combined_words(sections, 20)
    domains = top_domains(sections, 15)

    w_table = Table(box=box.SIMPLE, title="[bold white]🔤 Топ слов[/bold white]",
                    header_style="bold white")
    w_table.add_column("Слово",   style="white")
    w_table.add_column("×",       justify="right", style="yellow")
    for w, n in words:
        w_table.add_row(w, str(n))

    d_table = Table(box=box.SIMPLE, title="[bold cyan]🌐 Топ доменов[/bold cyan]",
                    header_style="bold cyan")
    d_table.add_column("Домен",   style="cyan")
    d_table.add_column("×",       justify="right", style="magenta")
    for d, n in domains:
        d_table.add_row(d, str(n))

    console.print(Columns([w_table, d_table]))


# ── Ollama ─────────────────────────────────────────────────────────────────────

def ensure_model(model: str) -> bool:
    try:
        available = [m.model for m in client.list().models]
        if any(model in m for m in available):
            return True
        console.print(f"{ts()} [yellow]⚠ Модель {model} не найдена, скачиваем...[/yellow]")
        for chunk in client.pull(model, stream=True):
            if hasattr(chunk, "status"):
                console.print(f"  {chunk.status}", end="\r")
        console.print()
        return True
    except Exception as e:
        console.print(f"{ts()} [red]❌ Ollama: {e}[/red]")
        return False


def run_ai(prompt: str, model: str, stream: bool = True) -> str:
    console.print(f"\n{ts()} [bold green]🤖 AI строит портрет...[/bold green] "
                  f"[dim]модель: {model} · темп: {temperature}[/dim]\n")
    console.print(Panel("", title="[bold green]ПСИХОЛОГИЧЕСКИЙ ПОРТРЕТ[/bold green]",
                         border_style="green", padding=(0, 1)))
    response = ""
    try:
        if stream:
            for chunk in client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                stream=True,
                options={"temperature": temperature, "num_ctx": num_ctx},
            ):
                token = chunk.message.content or ""
                console.print(token, end="", markup=False, highlight=False)
                response += token
            console.print("\n")
        else:
            r = client.chat(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                options={"temperature": temperature, "num_ctx": num_ctx},
            )
            response = r.message.content
            console.print(response)
    except Exception as e:
        console.print(f"\n[red]❌ {e}[/red]")
    return response


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="VK archive psychological portrait")
    ap.add_argument("--model",       default=selected_model)
    ap.add_argument("--no-stream",   action="store_true")
    ap.add_argument("--save",        action="store_true", help="Сохранить в logs/v8/")
    ap.add_argument("--prompt-only", action="store_true", help="Только показать промпт")
    args = ap.parse_args()

    print_header()
    console.print(f"{ts()} [cyan]Директория: [bold]{V8_DIR}[/bold][/cyan]\n")

    if not V8_DIR.exists():
        console.print("[red]❌ Директория v8/ не найдена[/red]")
        sys.exit(1)

    html_files = [f for f in sorted(V8_DIR.glob("*.html")) if f.name != "index.html"]

    # ── Парсинг ──────────────────────────────────────────────────────────────
    sections: dict[str, dict] = {}
    with Progress(
        SpinnerColumn(), TextColumn("[progress.description]{task.description}"),
        BarColumn(bar_width=28), TaskProgressColumn(), console=console,
    ) as prog:
        task = prog.add_task("[cyan]Парсинг...", total=len(html_files))
        for fpath in html_files:
            prog.update(task, description=f"[cyan]{fpath.name}")
            sections[fpath.name] = parse_section(fpath)
            prog.advance(task)

    console.print()

    # ── Таблицы статистики ───────────────────────────────────────────────────
    print_stats_table(sections)
    console.print()
    print_theme_table(sections)
    console.print()
    print_word_domain_tables(sections)
    console.print()
    console.print(f"{ts()} [dim]Временной диапазон: {timeline_hints(sections)}[/dim]")
    console.print()

    # ── Промпт ───────────────────────────────────────────────────────────────
    console.print(f"{ts()} [cyan]Составляю промпт...[/cyan]")
    prompt = compose_portrait_prompt(sections)
    console.print(f"{ts()} [dim]Промпт: {len(prompt):,} символов[/dim]\n")

    if args.prompt_only:
        console.print(Panel(
            prompt[:4000] + "\n[dim]… (усечено для preview)[/dim]",
            title="[yellow]Промпт[/yellow]", border_style="yellow",
        ))
        return

    # ── Ollama ───────────────────────────────────────────────────────────────
    if not ensure_model(args.model):
        sys.exit(1)

    response = run_ai(prompt, args.model, stream=not args.no_stream)

    if args.save and response:
        out = Path("logs") / "v8"
        out.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        (out / f"portrait_{stamp}.md").write_text(response, encoding="utf-8")
        (out / f"prompt_{stamp}.txt").write_text(prompt,   encoding="utf-8")
        console.print(f"{ts()} [green]✅ Сохранено в {out}[/green]")

    console.print(f"{ts()} [bold green]✅ Готово.[/bold green]")


if __name__ == "__main__":
    main()
