"""Shared layout primitives for the bilingual fixed-official-test reports."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from xml.sax.saxutils import escape
from zoneinfo import ZoneInfo
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas
from reportlab.platypus import Paragraph, Table, TableStyle

NAVY=colors.HexColor('#153149');TEAL=colors.HexColor('#087F8C');INK=colors.HexColor('#253649')
GRAY=colors.HexColor('#637487');PALE=colors.HexColor('#EFF5F8')
W,H=595.276,841.89
LEFT=RIGHT=44
CW=W-LEFT-RIGHT
styles={}
for name,size,leading,color in [('body',10.2,17,INK),('small',8.5,13,GRAY),
                              ('caption',8.5,13,GRAY),('table',8.3,12.5,INK),('thead',8.3,12.5,colors.white),
                              ('path',7.8,11,INK)]:
    styles[name]=ParagraphStyle(name,fontName='PathFont' if name=='path' else 'CN',fontSize=size,
                               leading=leading,textColor=color,wordWrap='CJK')

TOTAL_PAGES = 11
REPORT_TIME = datetime.now(timezone.utc).astimezone(ZoneInfo('Europe/London'))
page_records=[]
def P(text,style='body'):
    return Paragraph(text,styles[style])

def path_markup(value):
    """Wrap long Windows paths at directory boundaries, preserving every character."""
    lines, current = [], ''
    parts = value.split('\\')
    for index, part in enumerate(parts):
        token = part + ('\\' if index < len(parts) - 1 else '')
        if current and pdfmetrics.stringWidth(current + token, 'PathFont', 7.8) > CW:
            lines.append(current)
            current = ''
        current += token
    if current:
        lines.append(current)
    return '<br/>'.join(escape(line) for line in lines)

class Page:
    def __init__(self, number, label, title, subtitle=''):
        self.number, self.label, self.title = number, label, title
        c.setFillColor(NAVY)
        c.rect(0, H - 11, W, 11, fill=1, stroke=0)
        c.setFont('Arial', 8)
        c.setFillColor(TEAL)
        c.drawString(LEFT, H - 39, 'PIDG / FIXED OFFICIAL TEST: 50 VIEWS')
        c.setFont('CN', 8)
        c.setFillColor(GRAY)
        c.drawRightString(W - RIGHT, H - 39, label)
        title_size = min(22, 22 * CW / max(CW, pdfmetrics.stringWidth(title, 'CN-Bold', 22)))
        c.setFont('CN-Bold', title_size)
        c.setFillColor(NAVY)
        c.drawString(LEFT, H - 84, title)
        self.y = H - 104
        if subtitle:
            self.text(subtitle, 'small', gap=15)

    def check(self, height):
        if self.y - height < 58:
            raise RuntimeError(f'Page {self.number} overflow: y={self.y}, height={height}')

    def text(self, text, style='body', gap=10):
        obj = P(text, style)
        _, height = obj.wrap(CW, H)
        self.check(height + gap)
        obj.drawOn(c, LEFT, self.y - height)
        self.y -= height + gap

    def heading(self, text):
        self.check(29)
        c.setFillColor(TEAL)
        c.setFont('CN-Bold', 12)
        c.drawString(LEFT, self.y - 14, text)
        self.y -= 29

    def table(self, data, widths, highlight=None, size='table'):
        converted = [[P(escape(str(v)), 'thead' if i == 0 else size) for v in row]
                     for i, row in enumerate(data)]
        table = Table(converted, colWidths=widths, hAlign='LEFT')
        commands = [('BACKGROUND', (0, 0), (-1, 0), NAVY),
                    ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                    ('LEFTPADDING', (0, 0), (-1, -1), 7),
                    ('RIGHTPADDING', (0, 0), (-1, -1), 7),
                    ('TOPPADDING', (0, 0), (-1, -1), 7),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 7),
                    ('LINEBELOW', (0, 0), (-1, 0), 0.4, NAVY)]
        for i in range(1, len(data)):
            commands += [('BACKGROUND', (0, i), (-1, i), PALE if i % 2 else colors.white),
                         ('LINEBELOW', (0, i), (-1, i), .3, colors.HexColor('#D8E3EA'))]
        if highlight:
            commands.append(('BACKGROUND', (0, highlight), (-1, highlight), colors.HexColor('#DDF0EC')))
        table.setStyle(TableStyle(commands))
        _, height = table.wrap(CW, H)
        self.check(height + 12)
        table.drawOn(c, LEFT, self.y - height)
        self.y -= height + 12

    def image(self, path, height, caption=None):
        self.check(height + 8)
        c.drawImage(str(path), LEFT, self.y - height, width=CW, height=height,
                    preserveAspectRatio=True, anchor='c', mask='auto')
        self.y -= height + 8
        if caption:
            self.text(caption, 'caption', gap=10)

    def callout(self, text):
        obj = P(text)
        _, height = obj.wrap(CW - 24, H)
        self.check(height + 31)
        c.setFillColor(PALE)
        c.roundRect(LEFT, self.y - height - 20, CW, height + 20, 5, fill=1, stroke=0)
        c.setFillColor(TEAL)
        c.rect(LEFT, self.y - height - 20, 3, height + 20, fill=1, stroke=0)
        obj.drawOn(c, LEFT + 12, self.y - height - 10)
        self.y -= height + 31

    def finish(self):
        c.setStrokeColor(colors.HexColor('#D4E0E8'))
        c.setLineWidth(.5)
        c.line(LEFT, 44, W - RIGHT, 44)
        c.setFont('CN', 7.5)
        c.setFillColor(GRAY)
        c.drawString(LEFT, 30, 'Rubber Duck | 50k | official test: 50 views | '+REPORT_TIME.strftime('%Y-%m-%d'))
        c.setFont('Arial', 8)
        c.drawRightString(W - RIGHT, 30, f'{self.number:02d} / {TOTAL_PAGES}')
        page_records.append({'page': self.number, 'title': self.title, 'content_bottom': round(self.y, 2)})
        c.showPage()


