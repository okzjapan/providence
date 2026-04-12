"""WinTicket Betting Flow DOM Investigation (Deep).

投票シートの操作フロー（車番選択→追加→投票ボックス→金額→確定画面）を
ステップごとに記録する。実際に投票は行わない（最終確定は押さない）。

前提: investigate_winticket_dom.py を先に実行し、Chrome がログイン済みであること。

使い方:
  uv run python scripts/investigate_winticket_betting_flow.py
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT_DIR = Path("data/dom_investigation/betting_flow")
CDP_URL = "http://localhost:9222"


def save_step(page: Page, step_name: str, description: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = OUT_DIR / step_name
    page.screenshot(path=f"{prefix}.png", full_page=True)

    elements = []
    for el in page.query_selector_all(
        "button, a[href], input, select, [role='button'], [role='tab'], [role='checkbox'], "
        "[role='radio'], [role='dialog'], [class*='car'], [class*='number'], [class*='amount'], "
        "[class*='pin'], [class*='confirm'], [class*='add'], [class*='cart']"
    ):
        try:
            tag = el.evaluate("e => e.tagName")
            text = (el.inner_text() or "").strip()[:100]
            attrs = {
                "tag": tag,
                "text": text,
                "class": (el.get_attribute("class") or "")[:200],
                "role": el.get_attribute("role") or "",
                "data-testid": el.get_attribute("data-testid") or "",
                "type": el.get_attribute("type") or "",
                "name": el.get_attribute("name") or "",
                "placeholder": el.get_attribute("placeholder") or "",
                "aria-label": el.get_attribute("aria-label") or "",
                "aria-selected": el.get_attribute("aria-selected") or "",
                "value": el.get_attribute("value") or "",
                "maxlength": el.get_attribute("maxlength") or "",
            }
            elements.append(attrs)
        except Exception:
            pass

    Path(f"{prefix}_elements.json").write_text(
        json.dumps(elements, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"  [{step_name}] {description} ({len(elements)} elements)")


def find_and_click(page: Page, selectors: list[str], description: str) -> bool:
    for selector in selectors:
        try:
            el = page.query_selector(selector)
            if el and el.is_visible():
                el.click()
                print(f"  ✅ クリック成功: {selector} ({description})")
                page.wait_for_timeout(2000)
                return True
        except Exception:
            pass
    print(f"  ⚠ クリック失敗: {description} (試したセレクタ: {selectors})")
    return False


def investigate_betting_flow(page: Page) -> None:
    print(f"\n{'='*60}")
    print("WinTicket Betting Flow Investigation")
    print(f"{'='*60}\n")

    # Step 0: レースページに遷移（発売中のレースを自動選択）
    print("--- Step 0: 発売中レースに遷移 ---")
    page.goto("https://www.winticket.jp/autorace/", wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(5000)

    race_links = page.query_selector_all('a[href*="/autorace/"][href*="/racecard/"]')
    target_href = None
    for link in race_links:
        href = link.get_attribute("href") or ""
        text = (link.inner_text() or "").strip()
        if "投票受付中" in text and "/racecard/" in href:
            parts = href.split("/")
            if len(parts) >= 6:
                target_href = href
                print(f"  発売中レース発見: {href} → {text[:60]}")
                break

    if not target_href:
        print("  ❌ 発売中のレースが見つかりません")
        return

    full_url = f"https://www.winticket.jp{target_href}" if not target_href.startswith("http") else target_href
    page.goto(full_url, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(3000)
    save_step(page, "00_race_page", f"レースページ ({full_url})")

    # Step 1: 投票シートタブをクリック
    print("\n--- Step 1: 投票シートタブ ---")
    find_and_click(page, [
        'button[role="tab"]:has-text("投票シート")',
        'button:has-text("投票シート")',
    ], "投票シートタブ")
    save_step(page, "01_voting_sheet_tab", "投票シートタブ選択後")

    # Step 2: 賭式「単勝」を選択
    print("\n--- Step 2: 単勝を選択 ---")
    find_and_click(page, [
        'button[role="checkbox"]:has-text("単勝")',
        'button:has-text("単勝")',
    ], "単勝チェックボックス")
    save_step(page, "02_tansho_selected", "単勝選択後")

    # Step 3: 車番選択ボタンを調査（1号車をクリック）
    print("\n--- Step 3: 車番選択ボタン調査 ---")
    # 車番を表すボタン/要素を広く検索
    car_selectors_to_try = [
        # テキストで探す
        'button:has-text("1")',
        # aria-label で探す
        'button[aria-label*="1"]',
        # data-testid で探す
        '[data-testid*="car"]',
        '[data-testid*="number"]',
        '[data-testid*="runner"]',
        # クラス名で探す
        '[class*="CarNumber"]',
        '[class*="RunnerNumber"]',
        '[class*="BettingNumber"]',
        '[class*="EntryNumber"]',
    ]

    # まず全体像を記録
    all_buttons = page.query_selector_all("button")
    print(f"  ページ内の全ボタン: {len(all_buttons)} 件")
    for btn in all_buttons:
        text = (btn.inner_text() or "").strip()
        cls = (btn.get_attribute("class") or "")
        if text and len(text) <= 5:
            print(f"    <button> text='{text}' class=...{cls[-60:]}")

    # 出走表の行をクリック可能か調査
    print("\n  出走表の行/セル構造:")
    table_rows = page.query_selector_all("tr, [class*='Entry'], [class*='Runner'], [class*='row']")
    for row in table_rows[:10]:
        text = (row.inner_text() or "").strip()[:80].replace("\n", " | ")
        tag = row.evaluate("e => e.tagName")
        cls = (row.get_attribute("class") or "")[:80]
        clickable = row.evaluate("e => e.onclick !== null || e.closest('a, button') !== null")
        print(f"    <{tag}> clickable={clickable} text='{text}'")

    # 1着目の選択（投票シートのフォーメーション形式の場合）
    print("\n  投票シート内の数字ボタン/セレクタ:")
    number_elements = page.query_selector_all(
        "[class*='number'], [class*='Number'], [class*='pick'], [class*='Pick'], "
        "[class*='select'], [class*='Select'], [class*='car'], [class*='Car']"
    )
    for el in number_elements[:20]:
        text = (el.inner_text() or "").strip()[:20]
        tag = el.evaluate("e => e.tagName")
        cls = (el.get_attribute("class") or "")[:100]
        print(f"    <{tag}> text='{text}' class={cls}")

    save_step(page, "03_car_number_investigation", "車番選択エリア調査")

    # Step 4: 1号車を選択してみる（複数セレクタを試行）
    print("\n--- Step 4: 1号車を選択 ---")
    clicked = False
    # テーブル内の「1」をクリック（最初のもの）
    first_cells = page.query_selector_all("td, th")
    for cell in first_cells:
        text = (cell.inner_text() or "").strip()
        if text == "1":
            try:
                cell.click()
                print(f"  ✅ セルクリック: <td/th> text='1'")
                clicked = True
                page.wait_for_timeout(1000)
                break
            except Exception as e:
                print(f"  ⚠ セルクリック失敗: {e}")

    if not clicked:
        # ボタンとして「1」を探す
        for btn in all_buttons:
            text = (btn.inner_text() or "").strip()
            if text == "1":
                try:
                    btn.click()
                    print(f"  ✅ ボタンクリック: text='1'")
                    clicked = True
                    page.wait_for_timeout(1000)
                    break
                except Exception:
                    pass

    save_step(page, "04_after_car_select", "車番選択試行後")

    # Step 5: 「追加する」ボタンを探す
    print("\n--- Step 5: 追加するボタン ---")
    add_buttons = page.query_selector_all('button:has-text("追加"), button:has-text("セット"), button:has-text("add")')
    for btn in add_buttons:
        text = (btn.inner_text() or "").strip()
        visible = btn.is_visible()
        print(f"  追加ボタン: text='{text}' visible={visible}")

    if add_buttons:
        try:
            add_buttons[0].click()
            print("  ✅ 追加ボタンクリック")
            page.wait_for_timeout(2000)
        except Exception as e:
            print(f"  ⚠ 追加ボタンクリック失敗: {e}")

    save_step(page, "05_after_add", "追加ボタン後")

    # Step 6: 投票ボックスタブに切り替え
    print("\n--- Step 6: 投票ボックスタブ ---")
    find_and_click(page, [
        'button[role="tab"]:has-text("投票ボックス")',
        'button:has-text("投票ボックス")',
    ], "投票ボックスタブ")
    save_step(page, "06_voting_box", "投票ボックス")

    # Step 7: 金額入力欄と確定ボタンを調査
    print("\n--- Step 7: 金額入力・確定ボタン調査 ---")
    all_inputs = page.query_selector_all("input")
    for inp in all_inputs:
        attrs = {
            "type": inp.get_attribute("type") or "",
            "name": inp.get_attribute("name") or "",
            "placeholder": inp.get_attribute("placeholder") or "",
            "maxlength": inp.get_attribute("maxlength") or "",
            "value": inp.get_attribute("value") or "",
            "class": (inp.get_attribute("class") or "")[:80],
            "aria-label": inp.get_attribute("aria-label") or "",
        }
        visible = inp.is_visible()
        print(f"  <input> visible={visible} {attrs}")

    confirm_buttons = page.query_selector_all(
        'button:has-text("確定"), button:has-text("購入"), button:has-text("投票する")'
    )
    for btn in confirm_buttons:
        text = (btn.inner_text() or "").strip()[:40]
        visible = btn.is_visible()
        disabled = btn.get_attribute("disabled")
        print(f"  確定ボタン: text='{text}' visible={visible} disabled={disabled}")

    save_step(page, "07_amount_and_confirm", "金額入力・確定ボタン")

    # Step 8: ダイアログ/モーダルを調査（PIN入力は確定後に表示される可能性）
    print("\n--- Step 8: PIN入力関連 ---")
    pin_inputs = page.query_selector_all(
        'input[type="password"], input[type="tel"], input[maxlength="4"], '
        'input[maxlength="1"], [class*="pin"], [class*="Pin"], [class*="PIN"]'
    )
    for inp in pin_inputs:
        attrs = {
            "type": inp.get_attribute("type") or "",
            "maxlength": inp.get_attribute("maxlength") or "",
            "placeholder": inp.get_attribute("placeholder") or "",
            "class": (inp.get_attribute("class") or "")[:80],
        }
        print(f"  PIN候補: {attrs}")

    if not pin_inputs:
        print("  PIN入力欄は現時点で表示されていません（確定ボタン押下後に表示される可能性）")

    save_step(page, "08_final_state", "最終状態")

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
            print("Chrome が --remote-debugging-port=9222 で起動していることを確認してください。")
            return

        contexts = browser.contexts
        if not contexts or not contexts[0].pages:
            print("❌ ページが見つかりません")
            return

        page = None
        for pg in contexts[0].pages:
            if "winticket" in pg.url:
                page = pg
                break
        if page is None:
            page = contexts[0].pages[0]

        print(f"接続成功。対象ページ: {page.url}")
        investigate_betting_flow(page)


if __name__ == "__main__":
    main()
