"""WinTicket 自動投票テスト（安全機構付き）。

投票前にカートをクリアし、金額バリデーションを行う。

使い方:
  uv run python scripts/test_winticket_auto_bet.py
"""

from __future__ import annotations

import re
import time
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

from providence.config import get_settings

OUT_DIR = Path("data/dom_investigation/auto_bet_test_v2")
CDP_URL = "http://localhost:9222"

RACES = [
    {"race_no": 8, "car": 8, "label": "山陽R8"},
]

MEETING_ID = "2026040806"
TRACK_CODE = "5"
TRACK_SLUG = "sanyo"
BET_AMOUNT_UNITS = 1  # 1 = 100pt


def save_screenshot(page: Page, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    page.screenshot(path=str(OUT_DIR / f"{name}.png"), full_page=True)


def is_bet_type_active(page: Page, label: str) -> bool:
    btn = page.query_selector(f'button[role="checkbox"]:has-text("{label}")')
    if not btn:
        return False
    cls = btn.get_attribute("class") or ""
    return "enter-done" in cls or "appear-done" in cls


def clear_cart(page: Page, label: str) -> int:
    """投票ボックスの全アイテムを削除する。削除した件数を返す。"""
    page.click('button[role="tab"]:has-text("投票ボックス")')
    page.wait_for_timeout(2000)

    deleted = 0
    for _ in range(20):
        btns = page.query_selector_all('button[aria-label="削除する"]')
        visible = [b for b in btns if b.is_visible()]
        if not visible:
            break
        visible[0].click()
        page.wait_for_timeout(1000)
        deleted += 1

    if deleted:
        print(f"  [clear] {deleted} 件のアイテムを削除")
        save_screenshot(page, f"{label}_cleared")
    else:
        print("  [clear] カートは空でした")
    return deleted


def get_displayed_total(page: Page) -> int | None:
    """画面上の「投票合計 XXXpt」を取得する。"""
    body = page.inner_text("body")
    m = re.search(r"投票合計\s*(\d[\d,]*)pt", body)
    if m:
        return int(m.group(1).replace(",", ""))
    m = re.search(r"合計\s*(\d[\d,]*)pt", body)
    if m:
        return int(m.group(1).replace(",", ""))
    return None


def place_single_bet(page: Page, race_no: int, car: int, pin: str, label: str) -> bool:
    """1レースの単勝を投票する。"""
    expected_total = BET_AMOUNT_UNITS * 100
    url = f"https://www.winticket.jp/autorace/{TRACK_SLUG}/racecard/{MEETING_ID}/{TRACK_CODE}/{race_no}"

    print(f"\n{'='*50}")
    print(f"  {label}: {car}号車 単勝 {expected_total}pt")
    print(f"{'='*50}")

    # 1. レースページに遷移
    page.goto(url, wait_until="domcontentloaded", timeout=30000)
    page.wait_for_timeout(5000)
    print("  [1] レースページ読み込み完了")

    # 2. ★ カートクリア（既存アイテムを全削除）
    clear_cart(page, label)

    # 3. 投票シートタブに戻る
    page.click('button[role="tab"]:has-text("投票シート")')
    page.wait_for_timeout(2000)

    # 4. 賭式を単勝のみにする
    if is_bet_type_active(page, "3連単"):
        page.click('button[role="checkbox"]:has-text("3連単")')
        page.wait_for_timeout(500)
        print("  [4] 3連単を解除")

    if not is_bet_type_active(page, "単勝"):
        page.click('button[role="checkbox"]:has-text("単勝")')
        page.wait_for_timeout(1000)
        print("  [4] 単勝を選択")

    for bt in ["3連複", "2連単", "2連複", "複勝", "ワイド"]:
        if is_bet_type_active(page, bt):
            page.click(f'button[role="checkbox"]:has-text("{bt}")')
            page.wait_for_timeout(300)

    # 5. 車番を選択
    try:
        page.evaluate(f'document.querySelector(\'input[type="checkbox"][value="2:{car}"]\').click()')
        page.wait_for_timeout(2000)
        print(f"  [5] {car}号車を選択")
    except Exception as e:
        print(f"  ❌ 車番選択失敗: {e}")
        save_screenshot(page, f"{label}_error_car")
        return False

    # 6. 追加する
    add_btn = page.query_selector('button:has-text("追加する")')
    if not add_btn or not add_btn.is_visible():
        print("  ❌ 追加ボタンが見つかりません")
        save_screenshot(page, f"{label}_error_add")
        return False
    add_btn.click()
    page.wait_for_timeout(2000)
    print("  [6] 追加ボタンクリック")

    # 7. 投票ボックスタブ
    page.click('button[role="tab"]:has-text("投票ボックス")')
    page.wait_for_timeout(2000)

    # 8. 金額設定
    amount_input = page.query_selector('input[type="number"]')
    if amount_input:
        amount_input.fill(str(BET_AMOUNT_UNITS))
        page.wait_for_timeout(500)
        print(f"  [8] 金額: {BET_AMOUNT_UNITS} (= {expected_total}pt)")

    # 9. ★ 金額バリデーション
    page.wait_for_timeout(1000)
    displayed = get_displayed_total(page)
    print(f"  [9] バリデーション: 画面={displayed}pt / 意図={expected_total}pt")

    if displayed is not None and displayed != expected_total:
        print(f"  ❌ 金額不一致！ カートをクリアして中断します")
        clear_cart(page, f"{label}_mismatch")
        save_screenshot(page, f"{label}_error_mismatch")
        return False

    save_screenshot(page, f"{label}_09_validated")

    # 10. 確定する（1回目 → PIN ダイアログ）
    page.click('button:has-text("確定する")')
    page.wait_for_timeout(3000)
    print("  [10] 確定 → PIN ダイアログ")
    save_screenshot(page, f"{label}_10_pin")

    # 11. PIN 入力
    pin_input = page.query_selector('input[type="password"][maxlength="4"]')
    if not pin_input:
        print("  ❌ PIN 入力欄なし")
        save_screenshot(page, f"{label}_error_no_pin")
        return False
    pin_input.fill(pin)
    page.wait_for_timeout(500)
    print("  [11] PIN 入力完了")

    # 12. 確定する（2回目 → 投票確定）
    submit = page.query_selector('button[type="submit"]:has-text("確定する")')
    if not submit:
        print("  ❌ PIN 確定ボタンなし")
        save_screenshot(page, f"{label}_error_no_submit")
        return False
    submit.click()
    page.wait_for_timeout(5000)
    print("  [12] 投票送信")
    save_screenshot(page, f"{label}_12_result")

    # 13. 完了確認
    body = page.inner_text("body")
    if "受け付けました" in body or "受付" in body:
        print(f"  ✅ {label} 投票完了！")
        save_screenshot(page, f"{label}_13_success")
        return True
    else:
        print(f"  ⚠ 完了メッセージ未確認（スクリーンショットを確認）")
        save_screenshot(page, f"{label}_13_unknown")
        return True


def main() -> None:
    settings = get_settings()
    pin = settings.winticket_pin
    if not pin:
        print("❌ PROVIDENCE_WINTICKET_PIN が .env に未設定")
        return

    print(f"PIN: {'*' * len(pin)} ({len(pin)}桁)")
    print("\nChrome CDP に接続中...")

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
        print(f"\n⚠ 投票内容:")
        for r in RACES:
            print(f"  - {r['label']}: {r['car']}号車 単勝 {BET_AMOUNT_UNITS * 100}pt")

        if input("\n続行？ (y/n) > ").strip().lower() != "y":
            print("中断")
            return

        results = []
        for r in RACES:
            ok = place_single_bet(page, r["race_no"], r["car"], pin, r["label"])
            if not ok:
                print(f"\n  🔄 リトライ（1回目）...")
                time.sleep(3)
                ok = place_single_bet(page, r["race_no"], r["car"], pin, r["label"])
                if not ok:
                    print(f"\n  ⏭ 2回連続失敗 → {r['label']} をスキップ")
            results.append((r["label"], ok))
            time.sleep(2)

        print(f"\n{'='*50}")
        print("結果:")
        for label, ok in results:
            print(f"  {'✅' if ok else '❌'} {label}")
        print(f"{'='*50}")


if __name__ == "__main__":
    main()
