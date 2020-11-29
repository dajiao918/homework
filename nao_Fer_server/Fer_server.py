import socket


LOCAL_IP = '192.168.1.8'
PORT = 12345


def recognition():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # socket.AF_INET ipv4  socket.SOCK_STREAM tcp
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((LOCAL_IP, PORT))
    sock.listen(5)
    print('开始监听端口：')
    while True:
        c, addr = sock.accept()
        print('收到{}请求'.format(addr))
        infor = c.recv(1024)
        length, file_name = infor.split(b'|')
        if length and file_name:
            print('size:{},filename:{}'.format(length, file_name))
            c.sendall(b"got size and filename")
            got = 0
            data = b''
            while got < length:
                rec = c.recv(4096)
                data += rec
                got += len(rec)
            print('应该接收{}，实际接收{}'.format(length,len(data)))
            if data:
                newfile = open('image/image_rec.png','wb')
                newfile.write(data)
                newfile.close()
                c.sendall(b'got image')
                c.close()


if __name__ == '__main__':
    recognition()




