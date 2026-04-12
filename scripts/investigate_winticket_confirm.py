"""WinTicket 投票確定フロー調査。

山陽 R8 / 8号車 / 単勝 / 100pt を実際に投票し、
PIN 入力画面と確定後画面の DOM を記録する。

※ 実際に 100pt が消費されます。

使い方:
  uv run python scripts/investigate_winticket_confirm.py
"""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT_DIR = Path("data/dom_investigation/confirm_flow")
CDP_URL = "http://localhost:9222"

# 山陽 R8 のURL（先の調査で判明したパターン）
# /autorace/sanyo/racecard/{meeting_id}/5/{race_number}
RACE_URL = "https://www.winticket.jp/autorace/sanyo/racecard/2026040806/5/8"
TARGET_CAR = 8  # 8号車


def save_step(page: Page, step_name: str, description: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = OUT_DIR / step_name
    page.screenshot(path=f"{prefix}.png", full_page=True)

    elements = []
    for el in page.query_selector_all(
        "button, input, select, [role='button'], [role='tab'], [role='checkbox'], "
        "[role='dialog'], [class*='pin'], [class*='Pin'], [class*='confirm'], "
        "[class*='Confirm'], [class*='modal'], [class*='Modal'], [class*='dialog'], "
        "[class*='Dialog'], [class*='overlay'], [class*='Overlay']"
    ):
        try:
            tag = el.evaluate("e => e.tagName")
            text = (el.inner_text() or "").strip()[:80].replace("\n", " | ")
            visible = el.is_visible()
            elements.append({
                "tag": tag,
                "text": text,
                "class": (el.get_attribute("class") or "")[-100:],
                "role": el.get_attribute("role") or "",
                "type": el.get_attribute("type") or "",
                "value": el.get_attribute("value") or "",
                "name": el.get_attribute("name") or "",
                "placeholder": el.get_attribute("placeholder") or "",
                "aria-label": el.get_attribute("aria-label") or "",
                "maxlength": el.get_attribute("maxlength") or "",
                "disabled": el.get_attribute("disabled") or "",
                "visible": visible,
            })
        except Exception:
            pass

    Path(f"{prefix}_elements.json").write_text(
        json.dumps(elements, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"  💾 [{step_name}] {description} ({len(elements)} elements)")


def main() -> None:
    print("Chrome CDP に接続中...")
    with sync_playwright() as p:
        try:
            browser = p.chromium.connect_over_cdp(CDP_URL)
        except Exception as e:
            print(f"❌ {e}")
            return

        page = None
        for pg in browser.contexts[0].pages:
            if "winticket" in pg.url:
                page = pg
                break
        if not page:
            page = browser.contexts[0].pages[0]

        print(f"接続成功: {page.url}")
        print(f"\n⚠ 山陽 R8 / 8号車 / 単勝 / 100pt を投票します")
        print(f"  URL: {RACE_URL}")
        confirm = input("\n続行しますか？ (y/n) > ").strip().lower()
        if confirm != "y":
            print("中断しました。")
            return

        # Step 1: レースページに遷移
        print("\n--- Step 1: レースページに遷移 ---")
        page.goto(RACE_URL, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(5000)
        print(f"  URL: {page.url}")
        save_step(page, "01_race_page", "レースページ")

        # Step 2: 投票シートタブ
        print("\n--- Step 2: 投票シートタブ ---")
        page.click('button[role="tab"]:has-text("投票シート")')
        page.wait_for_timeout(2000)

        # Step 3: 3連単を解除（デフォルト選択されている場合）
        print("\n--- Step 3: 賭式を設定 ---")
        trifecta = page.query_selector('button[role="checkbox"]:has-text("3連単")')
        if trifecta:
            cls = trifecta.get_attribute("class") or ""
            if "enter-done" in cls or "appear-done" in cls:
                trifecta.click()
                page.wait_for_timeout(500)
                print("  3連単を解除")

        # 単勝を選択
        page.click('button[role="checkbox"]:has-text("単勝")')
        page.wait_for_timeout(1000)
        print("  単勝を選択")
        save_step(page, "02_tansho_selected", "単勝選択後")

        # Step 4: 8号車を選択
        print(f"\n--- Step 4: {TARGET_CAR}号車を選択 ---")
        checkbox_value = f"2:{TARGET_CAR}"
        page.evaluate(f'document.querySelector(\'input[type="checkbox"][value="{checkbox_value}"]\').click()')
        page.wait_for_timeout(2000)
        print(f"  {TARGET_CAR}号車のチェックボックスをクリック")
        save_step(page, "03_car_selected", f"{TARGET_CAR}号車選択後")

        # Step 5: 追加する
        print("\n--- Step 5: 追加する ---")
        page.click('button:has-text("追加する")')
        page.wait_for_timeout(2000)
        print("  追加ボタンクリック")
        save_step(page, "04_added", "追加後")

        # Step 6: 投票ボックスタブ
        print("\n--- Step 6: 投票ボックス ---")
        page.click('button[role="tab"]:has-text("投票ボックス")')
        page.wait_for_timeout(2000)
        save_step(page, "05_voting_box", "投票ボックス")

        # Step 7: 金額確認（デフォルトで 1 = 100pt が入っているはず）
        print("\n--- Step 7: 金額確認 ---")
        amount_input = page.query_selector('input[type="number"]')
        if amount_input:
            current_value = amount_input.get_attribute("value")
            print(f"  現在の金額入力値: {current_value}")
            if current_value != "1":
                amount_input.fill("1")
                page.wait_for_timeout(500)
                print("  金額を 1 (= 100pt) に設定")

        save_step(page, "06_amount_set", "金額設定後")

        # Step 8: 「確定する」をクリック！
        print("\n--- Step 8: 確定する ---")
        print("  ⚠ 「確定する」をクリックします...")
        page.click('button:has-text("確定する")')
        page.wait_for_timeout(3000)
        save_step(page, "07_after_confirm", "確定ボタン後")

        # Step 9: PIN 入力画面を調査
        print("\n--- Step 9: PIN 入力画面 ---")
        # ダイアログやモーダルが表示されるはず
        all_inputs = page.query_selector_all("input")
        for inp in all_inputs:
            visible = inp.is_visible()
            if not visible:
                continue
            attrs = {
                "type": inp.get_attribute("type") or "",
                "name": inp.get_attribute("name") or "",
                "placeholder": inp.get_attribute("placeholder") or "",
                "maxlength": inp.get_attribute("maxlength") or "",
                "value": inp.get_attribute("value") or "",
                "class": (inp.get_attribute("class") or "")[-80:],
                "aria-label": inp.get_attribute("aria-label") or "",
                "inputmode": inp.get_attribute("inputmode") or "",
            }
            print(f"  <input> visible=True {attrs}")

        # 全ボタンも確認
        for btn in page.query_selector_all("button"):
            if btn.is_visible():
                text = (btn.inner_text() or "").strip()[:40]
                if text:
                    print(f"  <button> text='{text}'")

        save_step(page, "08_pin_screen", "PIN入力画面")

        # Step 10: ここで一旦停止 → ユーザーに PIN 入力を依頼
        print("\n" + "=" * 50)
        print("PIN 入力画面のスクリーンショットを保存しました。")
        print()
        print("ブラウザで PIN (暗証番号) を手動で入力してください。")
        print("入力後、ターミナルで Enter を押してください。")
        print("=" * 50)
        input("\nPIN 入力完了後に Enter > ")

        page.wait_for_timeout(2000)
        save_step(page, "09_after_pin", "PIN入力後")

        # Step 11: 最終確定ボタンがあれば記録
        print("\n--- Step 11: 最終確定後 ---")
        # 投票完了画面を記録
        page.wait_for_timeout(3000)
        save_step(page, "10_final", "投票完了画面")

        print(f"\n{'='*50}")
        print(f"調査完了。結果は {OUT_DIR}/ に保存。")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
