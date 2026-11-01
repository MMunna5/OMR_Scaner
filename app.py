import io
import json
import os
import re

from flask import Flask, jsonify, request, send_file
from openpyxl import Workbook
from openpyxl.styles import Alignment, Font, PatternFill
from openpyxl.utils import get_column_letter

import omr
import omr_patch

app = Flask(__name__, static_folder='static', static_url_path='/static')
app.config['MAX_CONTENT_LENGTH'] = 40 * 1024 * 1024
MAX_PDF_PAGES = 30
MAX_EXPORT_ROWS = 1000


@app.errorhandler(413)
def request_too_large(_error):
    return jsonify(error='The uploaded file is too large. Maximum size is 40 MB.'), 413


def _valid_key(key, n):
    if not isinstance(key, list) or len(key) < n:
        return False
    for answer in key[:n]:
        choices = answer if isinstance(answer, list) else [answer]
        if not choices or any(type(x) is not int or x not in range(4) for x in choices):
            return False
        if len(set(choices)) != len(choices):
            return False
    return True


def _safe_excel_text(value, limit=200):
    value = str(value or '')[:limit]
    if value.lstrip().startswith(('=', '+', '-', '@')):
        value = "'" + value
    return value


@app.get('/')
def index():
    return send_file(os.path.join(app.static_folder, 'index.html'))


@app.get('/health')
def health():
    return 'ok'


@app.post('/api/scan')
def scan():
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify(error='No file was provided.'), 400
    try:
        n = int(request.form.get('n', 50))
        key = json.loads(request.form.get('key', '[]'))
        if not 1 <= n <= 50 or not _valid_key(key, n):
            raise ValueError
    except (TypeError, ValueError, json.JSONDecodeError):
        return jsonify(error='Invalid question count or answer key. Select at least one valid answer for every question.'), 400

    data = f.read()
    if not data:
        return jsonify(error='The uploaded file is empty.'), 400

    fname = os.path.basename(f.filename).replace('\\', '_')
    pages = omr.load_pages(fname, data)
    out, page_count = [], 0
    while True:
        try:
            label, img = next(pages)
        except StopIteration:
            break
        except Exception as e:
            app.logger.exception('Could not read uploaded file')
            out.append(dict(file=fname, error=str(e)[:300] or 'The file could not be read.'))
            break

        page_count += 1
        if page_count > MAX_PDF_PAGES:
            out.append(dict(file=fname, error=f'PDFs are limited to {MAX_PDF_PAGES} pages per upload. Split the PDF and try again.'))
            break
        try:
            r = omr.scan(img, n, key)
            r['file'] = label
        except omr.ScanError as e:
            r = dict(file=label, error=str(e))
        except Exception:
            app.logger.exception('scan failed')
            r = dict(file=label, error='The scan failed. Please try again with a clearer image.')
        out.append(r)

    if not out:
        out.append(dict(file=fname, error='No readable pages were found in this file.'))
    return jsonify(results=out)


@app.post('/api/export')
def export():
    d = request.get_json(silent=True)
    if not isinstance(d, dict):
        return jsonify(error='Invalid export request.'), 400
    try:
        n = int(d.get('n', 0))
        key, rows = d.get('key'), d.get('rows')
        if not 1 <= n <= 50 or not _valid_key(key, n):
            raise ValueError
        if not isinstance(rows, list) or not rows or len(rows) > MAX_EXPORT_ROWS:
            raise ValueError
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError
            if not isinstance(row.get('answers'), list) or not isinstance(row.get('status'), list):
                raise ValueError
            if len(row['answers']) < n or len(row['status']) < n:
                raise ValueError
            if any(s not in ('c', 'w', 'b', 'd') for s in row['status'][:n]):
                raise ValueError
            for field in ('correct', 'wrong', 'blank', 'double'):
                if type(row.get(field)) is not int or not 0 <= row[field] <= n:
                    raise ValueError
    except (TypeError, ValueError):
        return jsonify(error='The result data is incomplete or invalid. Please scan again and retry the export.'), 400

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
        correct = r['correct']
        ws.append([
            _safe_excel_text(r.get('roll'), 30), _safe_excel_text(r.get('name'), 100),
            correct, r['wrong'], r['blank'], r['double'], n, round(100 * correct / n, 1)
        ] + [_safe_excel_text(a, 8) for a in r['answers'][:n]])
        for i, status in enumerate(r['status'][:n]):
            cell = ws.cell(row=ws.max_row, column=9 + i)
            cell.alignment = Alignment(horizontal='center')
            if status == 'c':
                cell.fill = green
            elif status in ('w', 'b'):
                cell.fill = red
            elif status == 'd':
                cell.fill = amber

    for i in range(1, len(head) + 1):
        ws.column_dimensions[get_column_letter(i)].width = 16 if i == 2 else 8
    ws.freeze_panes = 'C2'

    key_sheet = wb.create_sheet('Answer Key')
    key_sheet.append(['Q', 'Answer'])
    for i in range(n):
        answers = key[i] if isinstance(key[i], list) else [key[i]]
        key_sheet.append([i + 1, ', '.join(omr.L[x] for x in answers)])

    buf = io.BytesIO()
    wb.save(buf)
    buf.seek(0)
    exam = re.sub(r'[\\\\/:*?"<>|\x00-\x1f]+', '-', str(d.get('exam') or 'omr-results')).strip(' .-')[:60]
    name = exam or 'omr-results'
    return send_file(buf, as_attachment=True, download_name=name + '.xlsx',
                     mimetype='application/vnd.openxmlformats-officedocument.spreadsheetml.sheet')


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=False)
