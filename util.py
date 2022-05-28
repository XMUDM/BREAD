def compare():
    dataset = 'bibsonomy'
    method = 'arrow'  #'bread', 'bread_gpu', 'rwbfs', 'hop'
    action = 'dynamic'  #'dynamic_hop'
    f = open("./queryFile/%s/%sQuery_%s_label_1k.txt" % (dataset, dataset, action))
    for line in f:
        a = line.split(',')[:-1]
    if method == 'arrow':
        f = open("./ARROW/result/%s_result_%s_%s_1k.txt" % (method, dataset, action))
    elif method == 'bread' or method == 'bfs' or method == 'rwbfs' or method == 'hop':
        f = open("./result/%s_result_%s_%s_1k.txt" % (method, dataset, action))
    elif method == 'bread_gpu':
        f = open("./gpu/result/%s_result_%s_%s_1k.txt" % (method, dataset, action))
    for line in f:
        b = line.split(',')[:-1]
    count = 0
    start = 0
    end = len(a)
    for i in range(start, end):
        if a[i] == b[i]:
            count += 1
    print("Accuracy: %.3f" % (count / 1000))

if __name__ == '__main__':
    compare()