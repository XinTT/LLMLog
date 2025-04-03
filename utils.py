
import difflib
import re



def post_processing(dataset,log,pred,keywords=[],variables=[],budget=-1):
    pred = pred.replace('<*,>','<*>').replace('<*/*>','<*>').replace('*<*>','<*>').replace('< *>','<*>').replace('(<*)>','(<*>)').replace('[<*]>','[<*>]')
    if '<>' not in log and '<>' in pred:
        pred = pred.replace('<>','<*>')
    pred = find_error_symbol(pred, '<*', '>')    # if '<>' not in log and '<>' in pred:
    if '\'' not in log and '\'' in pred:
        pred = pred.replace('\'','')
    if dataset == 'Android':
        if pred.endswith('(*,*)'):
            pred = pred[:-5]
        pred = pred.replace('(*)*','').replace('(*)',',').replace('(*,*)',',').replace('(null|<*>)','').replace('(<*>|<*>)','').replace('t<*>','<*>').replace('f<*>','<*>')
        if '(null)' not in log and '(null)' in pred:
            pred = pred.replace('(null)','')
        if '}}}' in log and '}}}' not in pred:
            pred = pred.replace('}}','}}}')
        if 'cancelNotification,' in log and 'cancelNotification:' in pred:
            pred = pred.replace('cancelNotification:','cancelNotification,')
        if 'size=' in log and 'size:' in pred:
            pred = pred.replace('size:','size=')
        if 'index:' in log and 'index=' in pred:
            pred = pred.replace('index=','index:')
        if 'id: ' in log and 'id=' in pred:
            pred = pred.replace('id=','id: ')
        if 'cancelNotificationLocked ' in pred and 'cancelNotificationLocked ' not in log:
            pred = pred.replace('cancelNotificationLocked ','cancelNotificationLocked,')
        if 'bnds=' in log and 'bnds=<*>' not in pred:
            if pred == '':
                pred = 'Skipping, withExcluded: false, tr.intent:Intent { flg=<*> cmp=<*> bnds=<*> }'
            else:
                pattern = re.compile(r'bnds=.+\}')
                # if pred == '':
                # print(f'error {log}')
                # print(pred)
                wrong_letters = pattern.findall(pred)
                # print(wrong_letters)
                pred = pred.replace(wrong_letters[0],'bnds=<*> }')
        if '(,' in pred and ')?' in pred:
            pred = pred.replace('(,',',').replace(')?','>')
    elif dataset == 'BGL':
        if pred.endswith('<*>:'):
            pred = pred[:-1]
        if '<*>: ' in pred:
            pred = pred.replace('<*>: ','<*>, ')
        pred = pred.replace(': <*><*>','')
        if 'program interrupt' == log:
            pred = log
        if 'after LOAD_MESSAGE ' not in log and 'after LOAD_MESSAGE ' in pred:
            pred = pred.replace('after LOAD_MESSAGE ','')
        if 'Input/output error' in log and 'No such file or directory' in pred:
            pred = pred.replace('No such file or directory','Input/output error')
        if ', bad message header: invalid cpu, type=<*>, cpu=<*>, index=<*>, total=<*>' in pred:
            pred = pred.replace(', bad message header: invalid cpu, type=<*>, cpu=<*>, index=<*>, total=<*>','')
        if 'floating point alignment exceptions' in log:
            pred = '<*> floating point alignment exceptions'
        pattern = re.compile('\.+')
                # findall return a list
        token = pattern.findall(log)
        token_replace = pattern.findall(pred)
        if len(token) > 0 and len(token_replace) > 0:
            pred =  pred.replace(token_replace[0],token[0])
        pred = pred.replace('<*>x<*>','<*>').replace('0x<*>','<*>').replace('(*,*)',',').replace('(null|<*>)','').replace('(<*>|<*>)','').replace('t<*>','<*>').replace('f<*>','<*>')
        for var in variables:
            if var in pred:
                
                pred = pred.replace(var,'<*>')
        if '<*> critical input' in pred:
            pred = pred.replace('<*> critical input','0 critical input')
        if 'critical input interrupts' in log: 
            return pred
    elif dataset == 'Thunderbird':
        pattern = re.compile('jA\S+\:')
        token = pattern.findall(pred)
        if len(token) != 0:
            pred = pred.replace(token[0],'<*>:')
        if 'PCI Interrupt Routing Table' in log:
            pred = 'PCI Interrupt Routing Table [<*>]'
        pattern = re.compile('#\d+')
        token = pattern.findall(log)
        if len(token) != 0:
            pred = pred.replace(token[0],'')
        if '[<*>)' in pred:
            pred = pred.replace('[<*>)','[<*>]')
            if pred.endswith(')') is False:
                pred += ')'
        elif '[0x<*>)' in pred:
            pred = pred.replace('[0x<*>)','[<*>]')
        
            if pred.endswith(')') is False:
                pred += ')'
        
        pattern = re.compile('0x[0-9A-Za-z]+')
        token = pattern.findall(pred)
        if len(token) != 0:
            pred = pred.replace(token[0],'<*>')
        if '(run-parts /etc/cron.hourly)' in log and '(run-parts /etc/cron.hourly)' not in pred:
            pred = pred.replace('(run-parts <*>)','(run-parts /etc/cron.hourly)')
        if 'network A_net' in log and 'network A_net' not in pred:
            pred = pred.replace('network <*>','network A_net')
        pred = pred.replace('sourceame','source name').replace('**','<*>').replace('reset:','reset').replace('[*]','[<*>]').replace('*:*:*','<*>:<*>:<*>').replace('*.*.*','<*>.<*>.<*>')
    elif dataset == 'Mac':
        if '::' in log and '<*>.<*>' in pred:
            pred = pred.replace('<*>.<*>','<*>::<*>')
        if '::' in log and '<=*/><*>' in pred:
            pred = pred.replace('<=*/><*>','<*>::<*>')
        pred = pred.replace('<?>','<*>').replace('0x<*>','<*>').replace('::<*> - <*> ','')
        if 'dev [<*>(<*>)]' in pred:
            pred = pred.replace('dev [<*>(<*>)]','dev [<*>,<*>]')
        pred = clean_string(log,pred)
        words = processpred(pred)
        for word in words:
            if word in variables or word == '0' or word.startswith('0x'):
                pred = pred.replace(word,'<*>')
        pred = pred.replace('tc<*>','<>')
        
        if 'path' in pred and 'error' in pred and 'origin' in pred:
            
            pred = convert_path(pred)
        if ']' in pred and 'file' in pred.lower():
        #     print(pred)
            pred = convert_file(log.strip(),pred.strip())
        elif '<<<< Boss >>>>' in pred:
            # print(pred)
            pred = convert_bracket(log.strip(),pred.strip())
        for item in processpred(pred):
            if item.isdigit():
                pred = pred.replace(item,'<*>')

        mac_words = log.split(' ')
        pred_words = pred.split(' ')
        if len(mac_words) == len(pred_words):
            for idx,word in enumerate(mac_words):
                if '<*>' in pred_words[idx] and word not in pred:
                    pred_words[idx] = word
            pred = ' '.join(pred_words)
        if pred.replace('<*>','') == log:
            pred = pred.replace('<*>','')
        # pred = log
        for var in variables:
            if var in pred:
                pred = pred.replace(var,'<*>')
    elif dataset == 'Proxifier':
        if len({"sent", "bytes","close","received","lifetime"}.intersection(keywords)) == 5:
            pred = '<*> close, <*> bytes<*>sent, <*> bytes<*>received, lifetime <*>'
            return pred
        else:
            pred = pred.replace('(*)','').replace('<<*>>','<*>').replace('*:','')
            if '<*> error :' in pred:
                pred = '<*>:<*> error : A connection request was canceled before the completion.'
            if 'open through proxy' in pred:
                redundant = pred.split(' ')[0]
                if redundant != '':
                    # for var in variables:
                    #     if redundant in var:
                    pred = pred.replace(redundant,'<*>:<*>')
            for var in variables:
                if var in pred:
                    pred = pred.replace(var,'<*>')
            for word in pred.split(' '):
                if '<*>' not in word and word not in log.strip():
                    f = 1
                    break
            pred = log
            for var in variables:
                if var in pred:
                    pred = pred.replace(var,'<*>')
            pred = pred.replace('<<*>>','<*>')
    elif dataset == 'OpenStack':
        f = 0
        for var in variables:
            if var in pred:
                f = 1
                break
        if len({"time","len", "status"}.intersection(keywords)) == 3:
            if 'GET' in pred:
                pred = '<*> \"GET <*>\" status: <*> len: <*> time: <*>.<*>'
            elif 'POST' in pred:
                pred = '<*> \"POST <*>\" status: <*> len: <*> time: <*>.<*>'
            elif 'DELETE' in pred:
                pred = '<*> \"DELETE <*>\" status: <*> len: <*> time: <*>.<*>'
    elif dataset == 'Windows':
        f = 0
        pattern = re.compile(r'<*>')
        results = pattern.findall(pred)
        if len(results) != pred.count('*'):
            # print(pred)
            pred = log
            for var in variables:
                if var in pred:
                    pred = pred.replace(var,'<*>')
        for var in variables:
            if var in pred:
                pred = pred.replace(var,'<*>')
        if '"<*>/' in pred and '"<*>/"' not in pred:
            pred = pred.replace('"<*>/','"<*>/"') 
    elif dataset == 'HDFS':
        f = 0
        for var in variables:
            if var in pred:
                f = 1
                break
        if 'blk_' in log and 'blk_' not in pred:
            f = 1
        for word in log.split():
            if word not in pred and word not in variables:
                f = 1
        # else:
        if f == 1 or pred == '':
            pred = log
            # variables.sorted()
            sorted_var = []
            a = sorted(list(variables),key=lambda x:len(x),reverse=True)
            
            for var in a:
                if var in pred:
                    pred = pred.replace(var,'<*>')
            return pred
            
        if '/<*>/<*>' in pred:
            pred = pred.replace('/<*>/<*>','/<*>/blk_<*>')
    elif dataset == 'Spark':
        f = 0
        for var in variables:
            if var in pred:
                f = 1
                break
        if f == 1 or pred == '':
            pred = log
            # variables.sorted()
            sorted_var = []
            a = sorted(list(variables),key=lambda x:len(x),reverse=True)
            
            for var in a:
                if var in pred:
                    pred = pred.replace(var,'<*>')
        
        pred = pred.replace('<*> KB','<*>').replace('<*> B','<*>')
    elif dataset == 'Apache':
        f = 0
        for var in variables:
            if var in pred:
                f = 1
                break
        if f == 1 or pred == '':
            pred = log
            # variables.sorted()
            sorted_var = []
            a = sorted(list(variables),key=lambda x:len(x),reverse=True)
            
            for var in a:
                if var in pred:
                    pred = pred.replace(var,'<*>')
    elif dataset == 'Zookeeper':
        f = 0
        for var in variables:
            if var in pred:
                f = 1
                break
        if f == 1 or pred == '':
            pred = log
            # variables.sorted()
            sorted_var = []
            a = sorted(list(variables),key=lambda x:len(x),reverse=True)
            
            for var in a:
                if var in pred:
                    pred = pred.replace(var,'<*>')
    elif dataset == 'Hadoop':
        pred = log
        a = sorted(list(variables),key=lambda x:len(x),reverse=True)
        # a = variables
        for var in a:
            if var in pred:
                pred = pred.replace(var,'<*>')
    elif dataset == 'OpenSSH':
        f = 0
        for var in variables:
            if var in pred:
                f = 1
                break
        for var in variables:
            if var in pred:
                f = 1
                break
        # else:
        if f == 1:
            pred = log
            for var in variables:
                if var in pred:
                    pred = pred.replace(var,'<*>')
        if '=*' in pred:
            pred = pred.replace('=*','=<*>')
        pattern = re.compile(r'(\s[a-zA-Z]+)(?=(\=\<\*\>))')
        results = pattern.findall(pred)
        if len(results) != 0:
            if 'pam_unix(sshd:auth): authentication failure; logname= uid=0 euid=0 tty=ssh ruser= rhost=103.99.0.122  user=ftp' == log:
                print(results)
            for result in results:
                if result[0]+'= ' in log:
                    pred = pred.replace(result[0]+'=<*>',result[0]+'= ')
        if pred == '':
            pred = 'input_userauth_request: invalid user <*> operator [preauth]'
            
        # if '/<*>/<*>' in pred:
        #     pred = pred.replace('/<*>/<*>','/<*>/blk_<*>')
        pred_words = []
        for word in pred.strip().split(' '):
            if word.replace('<*>','') in log or word == '<*>':
                pred_words.append(word)
        pred = ' '.join(pred_words)
    elif dataset == 'HPC':
        f = 0
        if 'storage' in log and 'storage' not in pred:
            f= 1
            
        if '<*>' not in pred and len(log.strip().split()) != pred.split() or pred.count('<*>') != len(variables) or '[<*>]' in pred:
            f = 1
        if log == 'ClusterFileSystem: There is no server for ServerFileSystem domain storage234':
            print(pred)
        variables = list(variables)
        variables = sorted(variables,key=lambda x:len(x),reverse=True)
        if log == 'Link error on broadcast tree Interconnect-1T00:00:2:0':
            print(pred)
            print(variables)
            # print(f)
        if f ==1:
            
            pred = log
            for var in variables:
                if var in pred:
                    
                    pred = pred.replace(var,'<*>')
            return pred
    if len(pred.split()) != len(log.split()) and '<*> <*>' in pred:
        pred = pred.replace('<*> <*>','<*>')
    matched,indices = find_replaced_characters(log.strip(),pred)
    if log == 'ClusterFileSystem: There is no server for ServerFileSystem domain storage234':
        print(f'{matched} {pred.strip()} {keywords}')
    if pred.find('<*>') != -1 and matched == []:
        pred = pred.replace('<*>','')
        for var in variables:
            if var in pred:
                pred = pred.replace(var,'<*>')
     
    cnt = 1
    duplicate = set()
    for item in matched:
        if item.strip() in keywords or (item.strip() not in variables and budget == 50) or item in {'true','false','null','null?true','null?false','end','begin','mapred.job.id','mapreduce.job.id'}:
            pred = replace_nth_occurrence(pred, '<*>', item, cnt)
            if dataset == 'HPC' and re.match(r'[A-Z]\d',item.strip()) is not None and re.match(r'[A-Z]\d',item.strip()).span()[1] == len(item.strip()):
                # print
                # print(pred)
                pred = pred.replace(item.strip(),item.strip()[0]+'<*>')
                # print(pred)
                
                duplicate.add(item)
                break
            if cnt > 1:
                cnt -= 1
        
        else:
            cnt += 1
    if {'true','false','null','null?true','null?false','end','begin'}.intersection(keywords) != {}:
        if 'cancelPeek:' in pred:
            if 'false' in log:
                pred = pred.replace('<*>',' false')
            else:
                pred = pred.replace('<*>',' true')
        if 'charging=' in pred:
            if 'false' in log:
                pred = pred.replace('charging= <*>',' charging=false')
            else:
                pred = pred.replace('charging= <*>',' charging=true')
        if 'restart:' in pred:
            if 'false' in log:
                pred = pred.replace('restart: <*>',' restart: false')
            else:
                pred = pred.replace('restart: <*>',' restart: true')
        if 'tag = <*>' in pred:
            
            start = pred.find('tag = <*>') 
            end = pred.find('pkg =') 
            if ',' in pred[start:end]:
                pred = pred.replace('tag = <*>',' tag = null')
            else:
                pred = pred.replace('tag = <*>',' tag = null,')
        if 'null?' in pred and 'null':
            if 'null?true' in log and 'null?true' not in pred:
                pred = pred.replace('null?','null?true')
            elif 'null?false' in log and 'null?false' not in pred:
                pred = pred.replace('null?','null?false')
        if 'pkg=<*>' in pred.replace(' ','') and 'pkg=<*>,' not in pred.replace(' ',''):
            pred = pred.replace('pkg =<*>',' pkg =<*>,').replace('pkg = <*>',' pkg = <*>,')
        if 'uid=<*>' in pred.replace(' ','') and 'uid=<*>,' not in pred.replace(' ',''):
            pred = pred.replace('uid = <*>',' uid = <*>,')
        if 'pid=<*>' in pred.replace(' ','') and 'pid=<*>,' not in pred.replace(' ',''):
            pred = pred.replace('pid = <*>',' pid = <*>,')
    if dataset == 'Mac':
        words = pred.split(' ')
        for word in words:
            if '0x' in word:
                pattern = re.compile(r'0x[A-Za-z0-9]+')
                # print(word)
                results = pattern.findall(word)  
                if len(results) > 0:              
                    pred = pred.replace(results[0],'<*>')
    return pred

def replace_nth_occurrence(a, b, c, n):
    start = 0
    count = 0

    while count < n:
        start = a.find(b, start)
        if start == -1:  # 如果没有找到匹配，返回原字符串
            return a
        count += 1
        start += len(b)  # 移动到下一个可能的匹配位置

    # 替换第 n 个 b 为 c
    return a[:start - len(b)] + c + a[start:]

def find_error_symbol(a, b, c):
    start = 0
    count = 0

    while start < len(a):
        start = a.find(b, start)
        if start == -1:  # 如果没有找到匹配，返回原字符串
            return a
        if a[start+2] != '>':
            a = a[:start+2] + '>' + a[start+2:]
            start += len(b) + 1  # 移动到下一个可能的匹配位置
        else:
            start += len(b)  # 移动到下一个可能的匹配位置

    # 替换第 n 个 b 为 c
    return a

def convert_bracket(log,pred):
    # [23:<*>:<*>.<*>]
    pattern1 = re.compile('\[[^a-zA-Z]+?\]\s')
    result1 = pattern1.findall(log)
    pattern2 = re.compile('\d+')
    results = []
    if len(result1) != 0:
        for item in result1:
            decimal_results = pattern2.findall(item)
            temp = item
            for decimal in decimal_results:
                
                results.append(decimal)
    # if '<*>]' in pred and '[<*>]' not in pred:
    #     pred = pred.replace('<*>]','[<*>]')
    pattern3 = re.compile('\[[0-9\.\:<>*]+?\]\s')
    result2 = pattern3.findall(pred)
    if results == []:
        # print(f'{pred} ')
    # if len(result2) > 10*len(results):
        return pred
    # print(f'{result2} {results} {result1}')
    for result in results:
        pred = pred.replace(result,'<*>')
    return pred

def convert_path(pred):
    pattern1 = re.compile('path\s=\s.+?\s(?=error)')
    pattern2 = re.compile('error\s=\s.+?\:')
    pattern3 = re.compile('origin\s=\s.+?$')
    result1 = pattern1.findall(pred)

    result2 = pattern2.findall(pred)
    result3 = pattern3.findall(pred)
    if len(result1) != 0:
        pred = pred.replace(result1[0],'path = <*> ')
    if len(result2) != 0:
        pred = pred.replace(result2[0],'error = <*>:')
    if len(result3) != 0:
        pred = pred.replace(result3[0],'origin = <*>')
    if 'type = <*>' in pred:
        pred = pred.replace('type = <*>','type = pid,')
    return pred

def convert_file(log,pred):
    if log.startswith('CCFile::captureLogRun Skipping current file Dir file'):
        pred = 'CCFile::captureLogRun Skipping current file Dir file[<*>-<*>-<*>_<*>,<*>,<*>.<*>]-<*>, Current File [<*>-<*>-<*>_<*>,<*>,<*>.<*>]-<*>'
    elif log.startswith('CCFile::copyFile fileName is'):
        pred = 'CCFile::copyFile fileName is [<*>-<*>-<*>_<*>,<*>,<*>.<*>]-<*>, source path:<*>, dest path:<*>'
    return pred

def clean_string(a,b):
    
    temp = ''
    diff_list = list(difflib.ndiff(a,b))
    pointer = 0
    # print(diff_list)
    for i, item in enumerate(diff_list):
        if item.startswith('+') is False and item.startswith('-') is False:
            temp += item[-1]
        elif item.startswith('-'):
            removed_string = ''
            if i < pointer:
                continue
            for idx in range(i,len(diff_list)):
                if diff_list[idx].startswith('-'):
                    removed_string += diff_list[idx][-1]
                    if idx + 1 == len(diff_list):
                        if i > 2 and diff_list[i-3] == '+ <' and diff_list[i-2] == '+ *' and diff_list[i-1] == '+ >':
                            pass
                        else:
                            temp += removed_string
                else:
                    
                    if diff_list[i-1].startswith(' ') and idx + 2 < len(diff_list) and diff_list[idx] == '+ <' and diff_list[idx+1] == '+ *' and diff_list[idx+2] == '+ >':
                        pass
                    elif diff_list[idx].startswith(' ') and i > 2 and diff_list[i-3] == '+ <' and diff_list[i-2] == '+ *' and diff_list[i-1] == '+ >':
                        pass
                    else:
                        temp += removed_string
                    pointer = idx
                    break
                    
                    

        else:
            if i + 2 < len(diff_list) and diff_list[i-1].startswith('+') is False and diff_list[i] == '+ <' and diff_list[i+1] == '+ *' and diff_list[i+2] == '+ >':
                temp += '<*>'
    return  temp

def find_replaced_characters(a, b, special_string='<*>'):
    replaced_indices = []  # 用于存储被替换的字符索引
    if '<*>:<*>:<*><*>:<*>' in b:
        b = b.replace('<*>:<*>:<*><*>:<*>','<*>:<*>:<*>:<*>:<*>:<*>')
    elif '<*><*>' in b:
        b = b.replace('<*><*>','<*>')
    special_length = len(special_string)
    indices,length,keywords = find_special_indices(b, special_string)
    # if a == 'Failed password for root from 183.62.140.253 port 38647 ssh2':
    #     print(keywords)
    i = 0
    j = 0
    k = 0
    # print(keywords)
    matched = []
    cnt = 0
    while i!= len(a):
        # print(f'{b.startswith(special_string,j)} {j}')
        if b.startswith(special_string,j):
            # temp = a[i,i+length[cnt]]
            temp = ''
            if keywords[cnt] != '':
                # print(i)
                f = 0
                for idx in range(i,len(a)):
                    if keywords[cnt] in temp:
                        
                        # print(cnt)
                        matched.append(temp.replace(keywords[cnt],''))
                        j += len(special_string)+len(keywords[cnt])
                        cnt += 1
                        # print(temp)
                        f = 1
                        break
                    temp += a[idx]
                    # print(temp)
                # print(f == 0)
                # print(keywords[cnt] in temp)
                if f == 0 and keywords[cnt] in temp:
                    matched.append(temp.replace(keywords[cnt],''))
            else:
                
                matched.append(a[i:len(a)])
                cnt += 1
            # print(f'{i} {len(a)} {temp} {cnt}')
            if cnt == len(keywords):
                break
            # print(f"{i}_{a[i]} {j}_{b[j]}")
            i += len(temp)
            
        else:
            i += 1 
            j += 1
    return matched,indices
                
def find_special_indices(b, special_string='<*>'):
    indices = []
    index = b.find(special_string)
    start_indices = []
    while index != -1:
        start_indices.append(index+len(special_string))
        indices.append(index)
        index = b.find(special_string, index + len(special_string))  # 从上一个找到的索引后继续查找
    # print(indices)
    # print(start_indices)
    temp = []
    for i in range(len(indices)):
        if i + 1 == len(indices):
            temp.append(len(b) - (indices[i]+len(special_string)))
        else:
            temp.append(indices[i+1] - (indices[i]+len(special_string)))
    keywords = []
    for i in range(len(temp)):
        keywords.append(b[start_indices[i]:start_indices[i]+temp[i]])
    # print(keywords)
    return indices,temp,keywords

def processpred(log):
    group = log
    group_initial = group
    words = []
    delimeter = r'[^A-Za-z0-9_@#$:/\=\*\|\?\-\.\<\>\s(\-\&\-)]'
    group = group.replace('<*>/<*>:<*>','').replace('<*>:<*>','').replace('<*>/<*>','').replace('<*>',' ')
    if group_initial == 'dnssd_clientstub ConnectToServer: connect()-> No of tries: <*>':
        words.extend(['dnssd_clientstub','ConnectToServer','connect()','->','No','of','tries'])
    # elif 'reduceResourceRequest:<memory:' in group_initial:
    #     words.extend(['reduceResourceRequest','memory','vCores'])

    # else:
    else:
        if '()' in group:
            group = re.sub(r'[\[\]\{\},]', ' ', group)
        else:
            group = re.sub(r'[\(\)\[\],]', ' ', group)
        group = re.sub(delimeter , '', group)
        # if log == '[CardDAVPlugin-ERROR] -getPrincipalInfo:[_controller supportsRequestCompressionAtURL:https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/] Error Domain=NSURLErrorDomain Code=-1001 \"The request timed out.\" UserInfo={NSUnderlyingError=0x7f9af3646900 {Error Domain=kCFErrorDomainCFNetwork Code=-1001 \"The request timed out.\" UserInfo={NSErrorFailingURLStringKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, NSErrorFailingURLKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, _kCFStreamErrorCodeKey=-2102, _kCFStreamErrorDomainKey=4, NSLocalizedDescription=The request timed out.}}, NSErrorFailingURLStringKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, NSErrorFailingURLKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, _kCFStreamErrorDomainKey=4, _kCFStreamErrorCodeKey=-2102, NSLocalizedDescription=The request timed out.}':
        #     print(f'{group}')
        #     print(group.split(' '))
        for word in group.split(' '):
            if len(word) == 1 and word.isnumeric() is False and word.isalpha() is False and word!='>' and word != '<':
                continue
            if word != '' and word !=' ' and word != '::':
                
                if word.endswith('.') and word.count('.') == 1:
                    words.append(word[:-1])
                elif word.startswith(':') and word.count(':') == 1 :
                    words.append(word[1:])
                
                elif '::' in word:
                    for w in word.split('::'):
                        if w != '' and w != ' ':
                            words.append(w)
                elif ':' in word and re.search(r'\d',word) is None:
                    
                    # if log == '2017-07-03 10:40:41.730 GoogleSoftwareUpdateAgent[33263/0x700000323000] [lvl=2] -[KSAgentApp performSelfUpdateWithEngine:] Checking for self update with Engine: <KSUpdateEngine:0x10062de70 ticketStore=<KSPersistentTicketStore:0x1005206e0 store=<KSKeyedPersistentStore:0x1005282c0 path=\"/Users/xpc/Library/Google/GoogleSoftwareUpdate/TicketStore/Keystone.ticketstore\" lockFile=<KSLockFile:0x100510480 path=\"/Users/xpc/Library/Google/GoogleSoftwareUpdate/TicketStore/Keystone.ticketstore.lock\" locked=NO > >> processor=<KSActionProcessor:0x10062e060 delegate=<KSUpdateEngine:0x10062de70> isProcessing=NO actionsCompleted=0 progress=0.00 errors=0 currentActionErrors=0 events=0 currentActionEvents=0 actionQueue=( ) > delegate=(null) serverInfoStore=<KSServerPrivateInfoStore:0x10062d2d0 path=\"/Users/xpc/Library/Google/GoogleSoftwareUpdate/Servers\"> errors=0 >':
                    #     print(f'test: {group}')
                    #     print(group.split(' '))
                    if '=' in word and word.endswith('=') is False:
                        word = word.replace(':','')
                        for w in word.split('='):
                            if w != '' and w != ' ':
                                words.append(w)
                    else:
                        for w in word.split(':'):
                            if w != '' and w != ' ':
                                words.append(w)
                elif word.count(':') == 1:
                    if log == '[CardDAVPlugin-ERROR] -getPrincipalInfo:[_controller supportsRequestCompressionAtURL:https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/] Error Domain=NSURLErrorDomain Code=-1001 \"The request timed out.\" UserInfo={NSUnderlyingError=0x7f9af3646900 {Error Domain=kCFErrorDomainCFNetwork Code=-1001 \"The request timed out.\" UserInfo={NSErrorFailingURLStringKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, NSErrorFailingURLKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, _kCFStreamErrorCodeKey=-2102, _kCFStreamErrorDomainKey=4, NSLocalizedDescription=The request timed out.}}, NSErrorFailingURLStringKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, NSErrorFailingURLKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, _kCFStreamErrorDomainKey=4, _kCFStreamErrorCodeKey=-2102, NSLocalizedDescription=The request timed out.}':
                        print(f'test: {group}')
                        print(group.split(' '))
                    if '=' in word and word.endswith('=') is False:
                        word = word.replace(':','')
                        for w in word.split('='):
                            if w != '' and w != ' ':
                                words.append(w)

                    else:
                        words.extend(word.split(':'))
                elif '=' in word and word.endswith('=') is False:
                    for w in word.split('='):
                        if w != '' and w != ' ':
                            words.append(w)
                    # words.extend(word.split('='))
                elif word.endswith('='):
                    word = word[:-1]
                    if '=' in word and word.endswith('=') is False:
                        for w in word.split('='):
                            if w != '' and w != ' ':
                                words.append(w)
                    else:
                        words.append(word)
                elif '|' in word:
                    for w in word.split('|'):
                        if w != '' and w != ' ':
                            words.append(w)
                elif '...'in word and word != '...':
                    words.append('...')
                    words.append(word.replace('...',''))
                else:
                    words.append(word)
        # if '***' in group_initial:
        #     words.append('***')
    if log == '[CardDAVPlugin-ERROR] -getPrincipalInfo:[_controller supportsRequestCompressionAtURL:https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/] Error Domain=NSURLErrorDomain Code=-1001 \"The request timed out.\" UserInfo={NSUnderlyingError=0x7f9af3646900 {Error Domain=kCFErrorDomainCFNetwork Code=-1001 \"The request timed out.\" UserInfo={NSErrorFailingURLStringKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, NSErrorFailingURLKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, _kCFStreamErrorCodeKey=-2102, _kCFStreamErrorDomainKey=4, NSLocalizedDescription=The request timed out.}}, NSErrorFailingURLStringKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, NSErrorFailingURLKey=https://13957525385%40163.com@p28-contacts.icloud.com/874161398/principal/, _kCFStreamErrorDomainKey=4, _kCFStreamErrorCodeKey=-2102, NSLocalizedDescription=The request timed out.}':
            print(f'{group}')
            print(group.split(' '))
            print(words)
    temp = []
    for item in words:
        if item != '' and item != ' ':
            temp.append(item)
    
    return temp