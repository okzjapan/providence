"""WinTicket Cart/Confirm Flow Investigation.

チェックボックスをクリックして買い目を追加し、
投票ボックス→金額入力→確定画面の DOM を記録する。
※「確定する」は押さない（投票しない）。

使い方:
  uv run python scripts/investigate_winticket_cart.py
"""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import Page, sync_playwright

OUT_DIR = Path("data/dom_investigation/cart_flow")
CDP_URL = "http://localhost:9222"


def save_step(page: Page, step_name: str, description: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    prefix = OUT_DIR / step_name
    page.screenshot(path=f"{prefix}.png", full_page=True)

    elements = []
    for el in page.query_selector_all(
        "button, input, select, [role='button'], [role='tab'], [role='checkbox'], "
        "[role='dialog'], [class*='amount'], [class*='Amount'], [class*='confirm'], "
        "[class*='Confirm'], [class*='pin'], [class*='Pin'], [class*='total'], [class*='Total']"
    ):
        try:
            tag = el.evaluate("e => e.tagName")
            text = (el.inner_text() or "").strip()[:100]
            elements.append({
                "tag": tag,
                "text": text,
                "class": (el.get_attribute("class") or "")[:200],
                "role": el.get_attribute("role") or "",
                "type": el.get_attribute("type") or "",
                "name": el.get_attribute("name") or "",
                "placeholder": el.get_attribute("placeholder") or "",
                "aria-label": el.get_attribute("aria-label") or "",
                "value": el.get_attribute("value") or "",
                "maxlength": el.get_attribute("maxlength") or "",
                "disabled": el.get_attribute("disabled") or "",
                "visible": el.is_visible(),
            })
        except Exception:
            pass

    Path(f"{prefix}_elements.json").write_text(
        json.dumps(elements, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"  [{step_name}] {description} ({len(elements)} elements)")


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

        print(f"接続成功: {page.url}\n")

        # 1. 発売中レースに遷移
        print("--- 1. レースページに遷移 ---")
        page.goto("https://www.winticket.jp/autorace/", wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(5000)

        race_links = page.query_selector_all('a[href*="/racecard/"]')
        target = None
        for link in race_links:
            text = (link.inner_text() or "")
            href = link.get_attribute("href") or ""
            if "投票受付中" in text and len(href.split("/")) >= 6:
                target = href
                break

        if not target:
            print("❌ 発売中レースなし")
            return

        url = f"https://www.winticket.jp{target}" if not target.startswith("http") else target
        page.goto(url, wait_until="domcontentloaded", timeout=30000)
        page.wait_for_timeout(3000)
        print(f"  レースURL: {page.url}")

        # 2. 投票シートタブ
        print("\n--- 2. 投票シートタブ ---")
        page.click('button[role="tab"]:has-text("投票シート")')
        page.wait_for_timeout(2000)

        # 3. まず全賭式をリセット（選択中の3連単等を解除してから単勝を選択）
        print("\n--- 3. 単勝を選択 ---")
        # 現在選択されている賭式を確認
        bet_buttons = page.query_selector_all('button[role="checkbox"]')
        for btn in bet_buttons:
            text = (btn.inner_text() or "").strip()
            aria_checked = btn.get_attribute("aria-checked") or ""
            cls = btn.get_attribute("class") or ""
            is_active = "enter-done" in cls or "appear-done" in cls or aria_checked == "true"
            print(f"  賭式: '{text}' active={is_active} aria-checked={aria_checked}")
            # 単勝以外が選択されていたら解除
            if is_active and text != "単勝":
                btn.click()
                page.wait_for_timeout(500)
                print(f"    → '{text}' を解除")

        # 単勝を選択（まだ選択されていなければ）
        tansho_btn = page.query_selector('button[role="checkbox"]:has-text("単勝")')
        if tansho_btn:
            cls = tansho_btn.get_attribute("class") or ""
            is_active = "enter-done" in cls or "appear-done" in cls
            if not is_active:
                tansho_btn.click()
                page.wait_for_timeout(1000)
                print("  ✅ 単勝を選択")
            else:
                print("  ✅ 単勝は既に選択済み")
        save_step(page, "00b_tansho_selected", "単勝選択後")

        # 確認: 選択後の賭式状態
        bet_buttons = page.query_selector_all('button[role="checkbox"]')
        for btn in bet_buttons:
            text = (btn.inner_text() or "").strip()
            cls = btn.get_attribute("class") or ""
            is_active = "enter-done" in cls or "appear-done" in cls
            if is_active:
                print(f"  ✓ 選択中の賭式: '{text}'")

        # 4. 1号車の1着目チェックボックスをクリック（JS click で確実に）
        print("\n--- 4. 1号車を選択 ---")
        # まずページ内のチェックボックスの状態を確認
        cbs = page.query_selector_all('input[type="checkbox"]')
        visible_cbs = []
        for cb in cbs:
            val = cb.get_attribute("value") or ""
            vis = cb.is_visible()
            if vis and val:
                visible_cbs.append((val, cb))
        print(f"  表示中のチェックボックス: {[v for v, _ in visible_cbs]}")

        # 1号車を選択
        clicked = False
        for val, cb in visible_cbs:
            if val == "2:1":
                page.evaluate('document.querySelector(\'input[type="checkbox"][value="2:1"]\').click()')
                print("  ✅ 1号車チェックボックスをクリック（JS click）")
                clicked = True
                page.wait_for_timeout(2000)
                break

        if not clicked:
            # value="2:1" が見つからない場合、最初の visible checkbox をクリック
            if visible_cbs:
                first_val, first_cb = visible_cbs[0]
                page.evaluate(f'document.querySelector(\'input[type="checkbox"][value="{first_val}"]\').click()')
                print(f"  ✅ フォールバック: value='{first_val}' をクリック（JS click）")
                page.wait_for_timeout(2000)

        save_step(page, "01_after_car_select", "車番選択後")

        # 5. 「追加する」「セットする」ボタンを探してクリック
        print("\n--- 5. 追加ボタン ---")
        add_clicked = False
        for selector in [
            'button:has-text("追加")',
            'button:has-text("セット")',
            'button:has-text("買い目に追加")',
            'button:has-text("投票に追加")',
        ]:
            btn = page.query_selector(selector)
            if btn and btn.is_visible():
                text = btn.inner_text().strip()[:40]
                btn.click()
                print(f"  ✅ クリック: '{text}'")
                add_clicked = True
                page.wait_for_timeout(2000)
                break

        if not add_clicked:
            # 全ボタンを表示して手がかりを得る
            print("  ⚠ 追加ボタンが見つかりません。全ボタン一覧:")
            for btn in page.query_selector_all("button"):
                text = (btn.inner_text() or "").strip()[:50]
                vis = btn.is_visible()
                if vis and text:
                    print(f"    '{text}' visible={vis}")

        save_step(page, "02_after_add", "追加ボタン後")

        # 6. 投票ボックスタブに切り替え
        print("\n--- 6. 投票ボックス ---")
        page.click('button[role="tab"]:has-text("投票ボックス")')
        page.wait_for_timeout(2000)
        save_step(page, "03_voting_box", "投票ボックス")

        # 7. 金額入力欄を調査
        print("\n--- 7. 金額入力・確定ボタン ---")
        for inp in page.query_selector_all("input"):
            visible = inp.is_visible()
            if not visible:
                continue
            attrs = {
                "type": inp.get_attribute("type") or "",
                "name": inp.get_attribute("name") or "",
                "placeholder": inp.get_attribute("placeholder") or "",
                "value": inp.get_attribute("value") or "",
                "maxlength": inp.get_attribute("maxlength") or "",
                "class": (inp.get_attribute("class") or "")[-60:],
                "aria-label": inp.get_attribute("aria-label") or "",
            }
            print(f"  <input> {attrs}")

        for btn in page.query_selector_all("button"):
            text = (btn.inner_text() or "").strip()[:40]
            visible = btn.is_visible()
            if visible and any(kw in text for kw in ["確定", "購入", "投票する", "入力"]):
                disabled = btn.get_attribute("disabled")
                print(f"  <button> text='{text}' disabled={disabled}")

        save_step(page, "04_amount_detail", "金額入力詳細")

        # 8. 金額「1」を入力してみる（最小100円 = "1"を入力）
        print("\n--- 8. 金額入力テスト ---")
        amount_input = None
        for inp in page.query_selector_all("input"):
            if inp.is_visible():
                inp_type = inp.get_attribute("type") or ""
                placeholder = inp.get_attribute("placeholder") or ""
                if inp_type in ("number", "tel", "text", "") and "pt" not in placeholder.lower():
                    amount_input = inp
                    break

        if amount_input:
            amount_input.fill("1")
            print("  ✅ 金額 '1' を入力")
            page.wait_for_timeout(1000)
        else:
            print("  ⚠ 金額入力欄が見つかりません")

        save_step(page, "05_amount_filled", "金額入力後")

        # 9. 「確定する」ボタンの状態（押さない！）
        print("\n--- 9. 確定ボタン（押しません） ---")
        for btn in page.query_selector_all("button"):
            text = (btn.inner_text() or "").strip()[:40]
            if btn.is_visible() and any(kw in text for kw in ["確定", "購入"]):
                disabled = btn.get_attribute("disabled")
                cls = (btn.get_attribute("class") or "")[-60:]
                print(f"  確定ボタン: text='{text}' disabled={disabled} class=...{cls}")

        save_step(page, "06_confirm_ready", "確定ボタン表示状態")

        print(f"\n{'='*60}")
        print(f"調査完了。結果は {OUT_DIR}/ に保存。")
        print("※ 確定ボタンは押していません。投票は行われていません。")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
