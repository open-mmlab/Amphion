
with open('data/eval_clean/text_note_short', 'r') as f, open('data/eval_clean/edit.short', 'w') as fw:
    for line in f.readlines():
        utt, phns = line.strip().split(maxsplit=1)
        if '(' not in phns:
            continue
        phns = phns.split()
        fw.write(utt + ' ')
        for i in range(len(phns)):
            if '(' in phns[i]:
                start = i
                fw.write(str(start)+' ')
            if ')' in phns[i]:
                end = i
                fw.write(str(end)+' ')
                break
        fw.write(' '.join(phns[start:end+1])[1:-1] + '\n')
