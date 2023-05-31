
# ENCAPSULATION EXAMPLE----------------------------------------------
# class Person:

#     def __init__(self, name, age, address):
#         self.__address = address
#         self.name = name
#         self.age = age

#     def get_name(self):
#         return self.name

#     def get_age(self):
#         return self.age

#     def get_address(self):
#         return self.__address

#     def has_reached_age_of_majority(self):
#         if self.age >= 18:
#             return True
#         else:
#             return False

#     def set_address(self, new_address):
#         if self.has_reached_age_of_majority() == True:
#             self.__address = new_address


# # POLYMORPHYSM EXAMPLE-----------------------------------------------
# class Shape:

#     def get_area(self):
#         return 0

# class Rectangle(Shape):

#     def __init__(self, width, height):
#         self.height = height
#         self.width = width

#     def get_area(self):
#         return self.width * self.height

# class Circle(Shape):

#     def __init__(self, radius):
#         self.radius = radius

#     def get_area(self):
#         return (self.radius ** 2) * 3.14

# rec = Rectangle(4, 5)
# circle = Circle(5)
# shapes = Shape.__subclasses__()
# for shape in shapes:
#     print(shape.get_area())
# # def make_area(shapes):
# #     area = []
# #     for shape in shapes:
# #         object_shape = shape()
# #         area.append(object_shape.get_area())
# #     return area
# # make_area(shapes)


# COMPOSITION(OBJECT AND OTHER COMPONENTS)------------------------
# class Book:

#     def __init__(self, title, author, publisher):
#         self.title = title
#         self.author = author
#         self.publisher = publisher

# class Library:

#     def __init__(self, title, author, publisher):
#         self.book = Book(title, author, publisher)
#         self.list = []
#         self.title = title
#         self.author = author
#         self.publisher = publisher
#         self.list.append(self.book.title)

#     def remove_book(self,removed_book):
#         self.list.remove(removed_book)

#     def add_book(self, added_book):
#         self.list.append(added_book.title)

#     def list_books(self):
#         return self.list

# lib1 = Library('fer', 'per', 'kaddy')
# lib2 = Library('lary', 'june', 'kaddy')
# lib3 = Library('sarry', 'manner', 'jason')
# lib3.add_book(lib2)
# lib3.add_book(lib1)
# print(lib2.list_books())


# class Engine:
#     def start(self):
#         print("Engine started.")

#     def stop(self):
#         print("Engine stopped.")

# class Car:
#     def __init__(self):
#         self.engine = Engine()

#     def start(self):
#         print("Starting car...")
#         self.engine.start()

#     def stop(self):
#         print("Stopping car...")
#         self.engine.stop()


# class Student:

#     breed = 'white'
#     def __init__(self, name, age):
#         self.name = name  # instance attribute
#         self.age = age # instance attribute

#     def set_breed(self, breeds):
#         self.breed = breeds


#     # @staticmethod
#     # def getobject():
#     #     return Student
#     @classmethod
#     def getobject(cls):
#         return cls('pary', 39)


# std1 = Student('ali', 25)

# std2 = std1.getobject()
# print(std1.name)
# print(std2.name)
# std2.set_breed('cacausian')
# print(std2.breed)

# INHERITANCE----------------------------------------------------
# class Employee:

#     raise_amt = 1.04

#     def __init__(self, first, last, pay):
#         self.first = first
#         self.last = last

#         # self.email = first + '.' + last + '@email.com'
#         self.pay = pay

#     # def get_fullname(self):
#     #     return '{} {}'.format(self.first,self.last)

#     @property
#     def email(self):
#         # self.email = first + '.' + last + '@email.com'
#         return '{}.{}@gmail.com'.format(self.first, self.last)

#     @email.setter
#     def email(self, mail):
#         first, lastandgmail, domain = mail.split('.')
#         last, domain = lastandgmail.split('@')
#         self.first = first
#         self.last = last

#     def apply_raise(self):
#         self.pay = int(self.pay * self.raise_amt)

#     # def __str__(self):
#     #     return '{} - {}'.format(self.get_fullname(), self.email)

#     # def __repr__(self):
#     #     return "Employee('{}', '{}', '{}')".format(self.first, self.last, self.pay)
#     @property
#     def get_fullname(self):
#         return '{} {}'.format(self.first,self.last)

#     @get_fullname.setter
#     def get_fullname(self, name):
#         first, last = name.split(' ')
#         self.first = first
#         self.last = last


# class Developer(Employee):

#     raise_amt = 1.08

#     def __init__(self, first, last, pay, prog_lang):
#         super().__init__(first, last, pay)
#         self.prog_lang = prog_lang


class Manager(Employee):

    # raise_amt = 1.10

    def __init__(self, first, last, pay, employees=None):
        super().__init__(first, last, pay)
        if employees is None:
            self.employees = []
        else:
            self.employees = employees

    def add_emp(self, emp):
        if emp not in self.employees:
            self.employees.append(emp)

    def remove_emp(self, emp):
        if emp in self.employees:
            self.employees.remove(emp)

    def print_emps(self):
        for emp in self.employees:
            print('--->', emp.get_fullname())
# print(help(Employee))


# dev_1 = Developer('amir', 'khojasteh', 2000, 'java')
# dev_2 = Developer('corey', 'schefer', 4000, 'python')

# mgr_1 = Manager('katy', 'perry', 5000, [dev_1])

# # print(mgr_1.print_emps())\
# mgr_1.print_emps()
# mgr_1.add_emp(dev_2)
# mgr_1.print_emps()
# # print(dev_1.email)
# # dev_1.apply_raise()
# # print(dev_1.get_fullname())
# # print(dev_1.prog_lang)

# emp_1 = Employee('amir', 'khojasteh', 2000)
# emp_2 = Employee('corey', 'schefer', 4000)

# emp_2.email = 'sory.nefer@gmail.com'
# emp_2.get_fullname = 'shory kefer'

# # get_fullname TAGHADOM BALATARY DARAD? CHERA?


# print(emp_2.get_fullname)
# print(emp_2.email)
# print(emp_2.pay)
# # print(emp_1.__str__())
# # print(emp_1.__repr__())

# TUPLE ANS LISTS EXAMPLE -----------------------------------------------------
# tuple1 = (1, 2, 3, 4, 'fery', 'pary')
# # print(tuple1[::2])
# for i in tuple1:
#     print(i)
# MUST REMOVE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
import pandas as pd
import urllib.request
from pathlib import Path


def download_and_extract():
    base_path = Path("path/datasets.tgz")
    if not Path.is_file("path"):
        Path("path").mkdir(parents=True)
        urllib.request.urlretrieve(url, filename=filename)


file = pd.read_csv()
# MUST REMOVE @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
