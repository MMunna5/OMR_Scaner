import io
import json
import os

from flask import Flask, jsonify, request, send_file
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import omr

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 40 * 1024 * 1024


@app.get('/')
def index():
    return send_file(os.path.join(app.static_folder, 'index.html'))


@app.get('/health')
def health():
    return 'ok'


@app.post('/api/scan')
def scan():
    """One uploaded file (photo or PDF) -> one result per page."""
    f = request.files.get('file')
    if not f:
        return jsonify(error='Kono file pawa jayni'), 400
    try:
        n = int(request.form.get('n', 50))
        key = json.loads(request.form.get('key', '[]'))
        if n not in (25, 30, 50) or len(key) < n:
            raise ValueError
    except ValueError:
        return jsonify(error='Question sonkha ba answer key thik nai'), 400
    out = []
    data, fname = f.read(), f.filename or 'upload'
    pages = omr.load_pages(fname, data)
    while True:
        try:
            label, img = next(pages)
        except StopIteration:
            break
        except Exception as e:  # unreadable file
            out.append(dict(file=fname, error=str(e) or 'File pora jayni'))
            break
        try:
            r = omr.scan(img, n, key)
            r['file'] = label
        except omr.ScanError as e:
            r = dict(file=label, error=str(e))
        except Exception:
            app.logger.exception('scan failed')
            r = dict(file=label, error='Scan korte somossa hoyeche. Abar clear photo diye try koro.')
        out.append(r)
    return jsonify(results=out)


@app.post('/api/export')
def export():
    d = request.get_json(force=True)
    n, key, rows = int(d['n']), d['key'], d['rows']
    wb = Workbook()
    ws = wb.active
    ws.title = 'Results'
    head = ['Roll', 'Name', 'Correct', 'Wrong', 'Blank', 'Double', 'Total', 'Percent'] + [f'Q{i}' for i in range(1, n + 1)]
    ws.append(head)
    for c in ws[1]:
        c.font = Font(bold=True, color='FFFFFF')
        c.fill = PatternFill('solid', fgColor='C62828')
        c.alignment = Alignment(horizontal='center')
    green, red, amber = (PatternFill('solid', fgColor=x) for x in ('C8E6C9', 'FFCDD2', 'FFE0B2'))
    for r in rows:
        ws.append([r['roll'], r['name'], r['correct'], r['wrong'], r['blank'], r['double'], n,
                   round(100 * r['correct'] / n, 1)] + r['answers'][:n])
        for i, s in enumerate(r['status'][:n]):
            cell = ws.cell(row=ws.max_row, column=9 + i)
            cell.alignment = Alignment(horizontal='center')
            if s == 'c':
                cell.fill = green
            elif s == 'w':
                cell.fill = red
            elif s == 'd':
                cell.fill = amber
    for i in range(1, len(head) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 16 if i == 2 else 8
    ws.freeze_panes = 'C2'
    k = wb.create_sheet('Answer Key')
    k.append(['Q', 'Answer'])
    for i in range(n):
        k.append([i + 1, omr.L[key[i]] if 0 <= key[i] < 4 else ''])
    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    name = (d.get('exam') or 'omr-results').strip()[:60].replace('/', '-') or 'omr-results'
    return send_file(buf, as_attachment=True, download_name=name + '.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
