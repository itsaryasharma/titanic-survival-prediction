"""Small presentation helpers (banner, result card, chart styling).

Only layout and styling live here; no ML logic.

Light/dark mode: colours for Streamlit's own widgets come from
.streamlit/config.toml ([theme.light] and [theme.dark]). The few custom HTML
elements below therefore avoid hard-coded white/black: they use translucent
backgrounds and inherit the text colour, so they read well on either theme.
"""
import html

import streamlit as st

ACCENT = "#2563EB"
# A mid-tone grey that keeps enough contrast on both a white and a near-black
# background; used for reference lines and neutral chart marks.
NEUTRAL = "#64748B"
PLOT_MARGIN = dict(l=10, r=10, t=10, b=10)


def inject_css(accent: str = ACCENT) -> None:
    st.markdown(
        f"""
        <style>
        .hero {{
            background: linear-gradient(135deg, {accent} 0%, #1F2937 130%);
            color: #FFFFFF; padding: 1.4rem 1.7rem; border-radius: 16px; margin-bottom: .4rem;
        }}
        .hero h1 {{ color: #FFFFFF; margin: 0 0 .35rem 0; font-size: 2rem; line-height: 1.2; padding: 0; }}
        .hero p {{ margin: 0; opacity: .93; font-size: 1.02rem; color: #FFFFFF; }}
        .badge {{
            display: inline-block; background: rgba(255,255,255,.18); color: #FFFFFF;
            padding: .15rem .65rem; border-radius: 999px; font-size: .78rem; margin: .7rem .35rem 0 0;
        }}
        .result-card {{
            border: 1px solid rgba(128,128,128,.28); border-left: 6px solid var(--card-color, {accent});
            border-radius: 12px; padding: 1rem 1.25rem; background: rgba(128,128,128,.07); margin-bottom: .8rem;
        }}
        .result-card .label {{ font-size: .8rem; text-transform: uppercase; letter-spacing: .06em; opacity: .75; }}
        .result-card .value {{ font-size: 1.9rem; font-weight: 700; line-height: 1.25; }}
        .result-card .sub {{ opacity: .85; font-size: .92rem; }}
        .footer {{ opacity: .65; font-size: .8rem; margin-top: 2rem; text-align: center; }}
        </style>
        """,
        unsafe_allow_html=True,
    )


def hero(title: str, subtitle: str, badges: list[str]) -> None:
    badge_html = "".join(f'<span class="badge">{html.escape(b)}</span>' for b in badges)
    st.markdown(
        f'<div class="hero"><h1>{html.escape(title)}</h1>'
        f"<p>{html.escape(subtitle)}</p>{badge_html}</div>",
        unsafe_allow_html=True,
    )
    st.caption(":material/contrast: **Light / dark mode:** open the **⋮** menu (top right) → **Settings** → "
               "**Theme**. Your choice is kept while you use the app.")


def result_card(label: str, value: str, sub: str = "", color: str = ACCENT) -> None:
    st.markdown(
        f'<div class="result-card" style="--card-color:{color}">'
        f'<div class="label">{html.escape(label)}</div>'
        f'<div class="value">{html.escape(value)}</div>'
        f'<div class="sub">{html.escape(sub)}</div></div>',
        unsafe_allow_html=True,
    )


def style_fig(fig, height: int, **layout):
    """Consistent chart styling that works in light and dark mode.

    Backgrounds are left transparent so Streamlit's active theme shows through,
    instead of forcing a white plot area.
    """
    fig.update_layout(height=height, margin=layout.pop("margin", PLOT_MARGIN),
                      paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", **layout)
    return fig


def footer(text: str) -> None:
    st.markdown(f'<div class="footer">{html.escape(text)}</div>', unsafe_allow_html=True)
