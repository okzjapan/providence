"""WinTicket DOM Investigation Script.

Chrome に CDP 接続し、WinTicket の投票フローの各ステップで
HTML とスクリーンショットを自動保存する。

使い方:
  1. Chrome を閉じる
  2. ターミナルで以下を実行:
     /Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
       --remote-debugging-port=9222 \
       --user-data-dir=$HOME/.chrome-winticket \
       https://www.winticket.jp/
  3. ブラウザで WinTicket にログイン（2FA 突破）
  4. このスクリプトを実行:
     uv run python scripts/investigate_winticket_dom.py
"""

from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT_DIR = Path("data/dom_investigation")
CDP_URL = "http://localhost:9222"


def save_step(page: Page, step_name: str, description: str) -> None:
    """1ステップ分の HTML とスクリーンショットを保存する。"""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = OUT_DIR / step_name

    page.screenshot(path=f"{prefix}.png", full_page=True)
    html = page.content()
    Path(f"{prefix}.html").write_text(html, encoding="utf-8")

    # 主要な要素のテキストとセレクタを抽出
    elements = []
    for el in page.query_selector_all("button, a, input, select, [role='button'], [role='tab']"):
        try:
            tag = el.evaluate("e => e.tagName")
            text = el.inner_text().strip()[:100] if el.inner_text() else ""
            classes = el.get_attribute("class") or ""
            role = el.get_attribute("role") or ""
            data_testid = el.get_attribute("data-testid") or ""
            href = el.get_attribute("href") or ""
            input_type = el.get_attribute("type") or ""
            placeholder = el.get_attribute("placeholder") or ""
            elements.append({
                "tag": tag,
                "text": text,
                "class": classes[:200],
                "role": role,
                "data-testid": data_testid,
                "href": href[:200],
                "type": input_type,
                "placeholder": placeholder,
            })
        except Exception:
            pass

    Path(f"{prefix}_elements.json").write_text(
        json.dumps(elements, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"  ✅ {step_name}: {description} ({len(elements)} elements)")


def investigate(page: Page) -> None:
    """WinTicket のページ構造を調査する。"""
    print(f"\n{'='*60}")
    print("WinTicket DOM Investigation")
    print(f"{'='*60}\n")

    # Step 1: 現在のページを記録
    current_url = page.url
    print(f"Current URL: {current_url}")
    save_step(page, "00_current_page", f"現在のページ ({current_url})")

    # Step 2: ログイン状態を確認
    print("\n--- ログイン状態チェック ---")
    balance_el = page.query_selector('[class*="balance"], [class*="point"], [class*="coin"]')
    if balance_el:
        print(f"  残高要素: {balance_el.inner_text()}")
    else:
        # ページ内のテキストからポイント/残高を探す
        body_text = page.inner_text("body")
        for line in body_text.split("\n"):
            if "ポイント" in line or "pt" in line.lower() or "残高" in line:
                print(f"  ポイント関連テキスト: {line.strip()[:100]}")
                break

    # Step 3: オートレースタブに遷移
    print("\n--- オートレースページに遷移 ---")
    page.goto("https://www.winticket.jp/autorace/", wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(5000)
    time.sleep(2)
    save_step(page, "01_autorace_top", "オートレーストップページ")

    # Step 4: 発売中のレース一覧を記録
    print("\n--- 発売中レースの構造 ---")
    race_links = page.query_selector_all('a[href*="/autorace/race/"], a[href*="/autorace/"]')
    print(f"  レースリンク数: {len(race_links)}")
    race_urls = []
    for link in race_links[:20]:
        href = link.get_attribute("href") or ""
        text = link.inner_text().strip()[:80]
        if "/race/" in href:
            race_urls.append(href)
            print(f"  {href}  →  {text}")

    if not race_urls:
        # レースリンクが見つからない場合、ページ全体のリンクを調査
        all_links = page.query_selector_all("a[href]")
        print(f"  全リンク数: {len(all_links)}")
        for link in all_links:
            href = link.get_attribute("href") or ""
            if "autorace" in href and "race" in href:
                text = link.inner_text().strip()[:80]
                race_urls.append(href)
                print(f"  [発見] {href}  →  {text}")

    # Step 5: 最初の発売中レースに遷移
    if race_urls:
        first_race_url = race_urls[0]
        if not first_race_url.startswith("http"):
            first_race_url = f"https://www.winticket.jp{first_race_url}"
        print(f"\n--- レースページに遷移: {first_race_url} ---")
        page.goto(first_race_url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(5000)
        time.sleep(2)
        save_step(page, "02_race_detail", "レース詳細ページ")

        # URL パターンを記録
        print(f"  レースURL: {page.url}")

        # 「投票シート」ボタンを探す
        print("\n--- 投票シートボタンを探索 ---")
        vote_buttons = []
        for selector in [
            'text="投票シート"', 'text="投票"', 'text="ベット"',
            '[class*="vote"]', '[class*="bet"]', '[class*="ticket"]',
            'button:has-text("投票")', 'a:has-text("投票")',
        ]:
            try:
                els = page.query_selector_all(selector)
                for el in els:
                    text = el.inner_text().strip()[:50]
                    tag = el.evaluate("e => e.tagName")
                    vote_buttons.append({"selector": selector, "text": text, "tag": tag})
                    print(f"  [発見] {selector} → <{tag}> '{text}'")
            except Exception:
                pass

        # 投票シートを開く
        if vote_buttons:
            print(f"\n--- 投票シートを開く ---")
            try:
                # テキストベースで最も確実なものをクリック
                for btn_info in vote_buttons:
                    if "投票" in btn_info["text"]:
                        page.click(btn_info["selector"], timeout=5000)
                        time.sleep(2)
                        save_step(page, "03_voting_sheet", "投票シート")
                        break
            except Exception as e:
                print(f"  ⚠ 投票シートを開けませんでした: {e}")
                save_step(page, "03_voting_sheet_error", "投票シートエラー")
        else:
            print("  ⚠ 投票シートボタンが見つかりません")
            # ページ内の全ボタンを記録
            all_buttons = page.query_selector_all("button")
            print(f"  ページ内の全ボタン ({len(all_buttons)} 件):")
            for btn in all_buttons[:30]:
                text = btn.inner_text().strip()[:60]
                if text:
                    print(f"    <button> '{text}'")

        # Step 6: 投票シート内の構造を調査
        print("\n--- 投票シート内の要素 ---")
        # 賭式選択
        for keyword in ["単勝", "ワイド", "3連単", "3連複", "2連単", "2連複"]:
            els = page.query_selector_all(f'text="{keyword}"')
            if els:
                for el in els:
                    tag = el.evaluate("e => e.tagName")
                    print(f"  賭式: '{keyword}' → <{tag}>")

        # 車番選択
        for num in range(1, 9):
            els = page.query_selector_all(f'text="{num}"')
            # 多すぎるのでカウントだけ
            pass

        # 金額入力
        input_els = page.query_selector_all('input[type="number"], input[type="text"], input[type="tel"]')
        for inp in input_els:
            placeholder = inp.get_attribute("placeholder") or ""
            name = inp.get_attribute("name") or ""
            print(f"  入力欄: name='{name}' placeholder='{placeholder}'")

        save_step(page, "04_voting_sheet_detail", "投票シート詳細")

        # 「オッズ」タブを探す（オッズから投票する方法も調査）
        print("\n--- オッズタブを調査 ---")
        for selector in ['text="オッズ"', '[class*="odds"]', 'a:has-text("オッズ")']:
            try:
                els = page.query_selector_all(selector)
                for el in els:
                    text = el.inner_text().strip()[:50]
                    print(f"  [発見] {selector} → '{text}'")
            except Exception:
                pass

    # Step 7: PIN 入力欄を探す（確定ボタン押下なしで）
    print("\n--- PIN 入力欄の構造 ---")
    pin_inputs = page.query_selector_all('input[type="password"], input[type="tel"], input[maxlength="4"]')
    for inp in pin_inputs:
        attrs = {
            "type": inp.get_attribute("type"),
            "name": inp.get_attribute("name"),
            "maxlength": inp.get_attribute("maxlength"),
            "placeholder": inp.get_attribute("placeholder"),
            "class": (inp.get_attribute("class") or "")[:100],
        }
        print(f"  PIN候補: {attrs}")

    # 完了
    print(f"\n{'='*60}")
    print(f"調査完了。結果は {OUT_DIR}/ に保存されています。")
    print(f"{'='*60}")


def main() -> None:
    print("Chrome CDP に接続中...")
    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(CDP_URL)
        except Exception as e:
            print(f"\n❌ Chrome に接続できません: {e}")
            print("\n以下の手順で Chrome を起動してください:")
            print("  1. 全ての Chrome ウィンドウを閉じる")
            print("  2. ターミナルで以下を実行:")
            print('     /Applications/Google\\ Chrome.app/Contents/MacOS/Google\\ Chrome \\')
            print("       --remote-debugging-port=9222 \\")
            print("       --user-data-dir=$HOME/.chrome-winticket \\")
            print("       https://www.winticket.jp/")
            print("  3. ブラウザで WinTicket にログイン")
            print("  4. このスクリプトを再実行")
            return

        contexts = browser.contexts
        if not contexts:
            print("❌ ブラウザコンテキストが見つかりません")
            return

        pages = contexts[0].pages
        if not pages:
            print("❌ タブが見つかりません")
            return

        print(f"接続成功。{len(pages)} タブを検出。")
        for i, pg in enumerate(pages):
            print(f"  タブ {i}: {pg.url}")

        # WinTicket のタブを探す
        target_page = None
        for pg in pages:
            if "winticket" in pg.url:
                target_page = pg
                break

        if target_page is None:
            target_page = pages[0]
            print(f"WinTicket タブが見つからないため、最初のタブを使用: {target_page.url}")

        investigate(target_page)


if __name__ == "__main__":
    main()
