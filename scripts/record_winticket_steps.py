"""WinTicket Step Recorder.

ユーザーが手動で WinTicket を操作し、各ステップで Enter を押すと
DOM 状態（スクリーンショット + 要素リスト）を保存する。

使い方:
  1. Chrome で WinTicket のレースページを開いておく
  2. このスクリプトを実行:
     uv run python scripts/record_winticket_steps.py
  3. ブラウザで 1 操作するごとに、ターミナルで Enter を押す
  4. 'q' + Enter で終了
"""

from __future__ import annotations

import json
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT_DIR = Path("data/dom_investigation/manual_record")
CDP_URL = "http://localhost:9222"


def save_snapshot(page, step_num: int, label: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    name = f"{step_num:02d}_{label}"
    prefix = OUT_DIR / name

    page.screenshot(path=f"{prefix}.png", full_page=True)

    elements = []
    for el in page.query_selector_all(
        "button, input, select, [role='button'], [role='tab'], [role='checkbox']"
    ):
        try:
            tag = el.evaluate("e => e.tagName")
            text = (el.inner_text() or "").strip()[:80].replace("\n", " | ")
            visible = el.is_visible()
            if not visible:
                continue
            elements.append({
                "tag": tag,
                "text": text,
                "class": (el.get_attribute("class") or "")[-80:],
                "role": el.get_attribute("role") or "",
                "type": el.get_attribute("type") or "",
                "value": el.get_attribute("value") or "",
                "aria-label": el.get_attribute("aria-label") or "",
                "aria-checked": el.get_attribute("aria-checked") or "",
                "disabled": el.get_attribute("disabled") or "",
                "placeholder": el.get_attribute("placeholder") or "",
                "maxlength": el.get_attribute("maxlength") or "",
            })
        except Exception:
            pass

    Path(f"{prefix}_elements.json").write_text(
        json.dumps(elements, ensure_ascii=False, indent=2), encoding="utf-8",
    )
    print(f"  💾 保存: {name} ({len(elements)} 要素)\n")


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
        print()
        print("=" * 50)
        print("手順:")
        print("  1. ブラウザで操作を 1 ステップ行う")
        print("  2. ターミナルに戻って Enter を押す")
        print("  3. → スクリーンショット + DOM が保存される")
        print("  4. 繰り返し。'q' で終了")
        print()
        print("推奨操作手順:")
        print("  ① レースページで「投票シート」タブをクリック")
        print("  ② 「3連単」を解除（クリックして外す）")
        print("  ③ 「単勝」を選択")
        print("  ④ 1号車のチェックボックスをクリック")
        print("  ⑤ 「追加する」ボタンをクリック")
        print("  ⑥ 「投票ボックス」タブをクリック")
        print("  ⑦ 金額欄に「1」を入力")
        print("  ⑧ ここでストップ（確定は押さない！）")
        print("=" * 50)
        print()

        step = 0

        # 初期状態を保存
        save_snapshot(page, step, "initial")
        step += 1

        while True:
            user_input = input(f"Step {step}: 操作が完了したら Enter（'q' で終了）> ").strip()
            if user_input.lower() == "q":
                break

            label = input(f"  ラベル（例: tansho_selected）> ").strip() or f"step{step}"
            save_snapshot(page, step, label)
            step += 1

        print(f"\n記録完了。{step} ステップ保存。")
        print(f"結果: {OUT_DIR}/")


if __name__ == "__main__":
    main()
